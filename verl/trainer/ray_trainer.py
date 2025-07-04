import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .core_algos import (
    FixedKLController,
    KLController,
    _convert_and_check_hit,
    _parse_coords_from_response,
    compute_kl,
    get_kl_controller,
)
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics


class Role(IntEnum):
    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld
    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics

from .core_algos import AdvantageEstimator

def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0, config: PPOConfig = None):
    token_level_rewards = data.batch.get("token_level_rewards") 
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch.get("uid")

    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.MGRPO_V:
        advantages, returns = core_algos.compute_mgrpo_v_advantage(
            decoded_responses=data.non_tensor_batch["decoded_responses"],
            gt_bboxes_xywh=data.batch["gt_bboxes_xywh"],
            image_dims=data.batch["image_dims"],
            response_mask=data.batch["response_mask"],
            trajectory_ids=data.batch["trajectory_ids"],
            turn_ids=data.batch["turn_ids"],
            w_outcome=config.algorithm.w_outcome,
            w_prog=config.algorithm.w_prog,
        )
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        self.use_critic = config.algorithm.adv_estimator == AdvantageEstimator.GAE

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")
        
        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")
        if (config.data.rollout_batch_size * config.worker.rollout.n) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError("Rollout batch size * rollout.n must be divisible by actor micro batch size for experience.")
        if self.use_critic and (config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0):
            raise ValueError("Rollout batch size must be divisible by critic global batch size.")
        if (config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO) and config.worker.rollout.n == 1):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")
        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref")
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic")
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
        if self.use_reward_model:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward")
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()
        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step
        remove_obsolete_ckpt(self.config.trainer.save_checkpoint_path, self.global_step, self.best_global_step, self.config.trainer.save_limit)
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)
        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)
        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)
        checkpointer_tracker_info = {"best_global_step": self.best_global_step, "best_val_reward_score": round(self.best_val_reward_score, 4), "last_global_step": self.global_step, "last_actor_path": os.path.abspath(actor_path)}
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return
        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")
        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)
        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(self, samples: List) -> None:
        if self.config.trainer.val_generations_to_log <= 0:
            return
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        
        if self.config.algorithm.adv_estimator == "mgrpo_v":
            total_samples = 0
            correct_hits = 0
            
            for batch_dict in self.val_dataloader:
                val_batch = DataProto.from_single_dict(batch_dict)
                
                gen_batch = val_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "pixel_values", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )
                
                gen_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
                decoded_responses = gen_output.non_tensor_batch["decoded_responses"]

                gt_bboxes = batch_dict["gt_bboxes_xywh"]
                image_dims = batch_dict["image_dims"]

                for i in range(len(decoded_responses)):
                    total_samples += 1
                    try:
                        pred_coords = _parse_coords_from_response(decoded_responses[i])
                        img_w, img_h = image_dims[i].tolist()
                        gt_bbox = gt_bboxes[i].tolist()
                        
                        if _convert_and_check_hit(pred_coords, gt_bbox, img_w, img_h):
                            correct_hits += 1
                    except Exception:
                        continue
            
            hit_rate = (correct_hits / total_samples) if total_samples > 0 else 0.0
            self.val_reward_score = hit_rate
            val_metrics = {"val/hit_rate": hit_rate}

        else: 
            # ... (original validation logic for other algorithms) ...
            pass

        self.actor_rollout_ref_wg.release_rollout_engine()
        print("Finish validation.")
        return val_metrics

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)


    def _make_batch_data(self, metrics: Dict[str, Any]) -> DataProto:
        batch = None
        print("Start generating batch...")
        while True:
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            new_batch: DataProto = DataProto.from_single_dict(batch_dict)
            
            repeated_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids", "pixel_values"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )

            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)
            
            repeated_batch.batch["responses"] = gen_batch_output.batch["responses"]
            if "log_probs" in gen_batch_output.batch:
                repeated_batch.batch["log_probs"] = gen_batch_output.batch["log_probs"]

            final_batch = repeated_batch

            decoded_list = self.tokenizer.batch_decode(final_batch.batch["responses"], skip_special_tokens=True)
            final_batch.non_tensor_batch["decoded_responses"] = np.array(decoded_list, dtype=object)

            batch = DataProto.concat([batch, final_batch]) if batch is not None else final_batch
            
            current_batch_size = len(batch) // self.config.worker.rollout.n
            if current_batch_size >= self.config.data.rollout_batch_size:
                print("Finish generating batch.")
                return batch[:self.config.data.rollout_batch_size * self.config.worker.rollout.n]


    def fit(self):
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[Dict[str, Any]] = None
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        if self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1
            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()
                
                if "responses" in batch.batch and "response_mask" not in batch.batch:
                    response_mask = (batch.batch["responses"] != self.tokenizer.pad_token_id).long()
                    batch.batch["response_mask"] = response_mask
                
                self._balance_batch(batch, metrics=metrics)
                
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)
                
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)
                
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if self.config.algorithm.adv_estimator != "mgrpo_v":
                        if "token_level_scores" not in batch.batch:
                            reward_ref = self.reward_fn.compute_reward.remote(batch)
                            reward_tensor, reward_metrics = ray.get(reward_ref)
                            batch.batch["token_level_scores"] = reward_tensor
                            reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                            metrics.update(reward_metrics)
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        config=self.config,
                    )

                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)
                
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)
                
                if (self.config.trainer.val_freq > 0 and self.global_step % self.config.trainer.val_freq == 0):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)
                
                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        if (val_metrics is None or self.config.trainer.val_freq <= 0 or self.global_step % self.config.trainer.val_freq != 0):
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
        print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")
        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()