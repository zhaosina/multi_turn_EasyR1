# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# run_mgrpo_v_training.py
import json
import ray
from omegaconf import OmegaConf
import time
import os
import logging

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .config import PPOConfig

# 导入数据加载函数
from .data_loader import create_dataloader
from .data_loader import create_mgrpo_v_dataloader

from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        log_dir = "/home/zhq/workdir/GUI/EasyR1/verl/trainer/train_log"
        os.makedirs(log_dir, exist_ok=True)
        
        log_filename = f"mgrpo_v_run_{time.strftime('%Y%m%d-%H%M%S')}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        # 配置此 Actor 的根记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_filepath,
            filemode='w',
            force=True  # [关键] force=True 确保可以覆盖 Ray 可能设置的任何默认配置
        )
        # 打印配置
        print("--- Training Configuration ---")
        print(json.dumps(config.to_dict(), indent=2))
        print("------------------------------------")

        # 加载 tokenizer 和 processor
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # WorkerGroup 类和角色映射
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic:         ray.remote(FSDPWorker),
        }

        # 资源池配置
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic:          global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        # 根据 adv_estimator 决定是否创建外部 RewardManager
        reward_fn = None
        val_reward_fn = None
        if config.algorithm.adv_estimator != "mgrpo_v":
            print(f"Initializing external reward manager for estimator: '{config.algorithm.adv_estimator}'")
            if config.worker.reward.reward_type == "sequential":
                RewardManager = SequentialFunctionRewardManager
            elif config.worker.reward.reward_type == "batch":
                RewardManager = BatchFunctionRewardManager
            else:
                raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")
            RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
            reward_fn     = RemoteRewardManager.remote(config.worker.reward, tokenizer)
            val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        else:
            print("Algorithm is 'mgrpo_v'. Skipping external reward manager initialization.")

        # =================================================================
        # 根据算法类型选择正确的数据加载函数
        # =================================================================
        if config.algorithm.adv_estimator == "mgrpo_v":
            print("Creating MGRPO-V dataloaders using 'create_mgrpo_v_dataloader'...")
            train_dataloader, val_dataloader = create_mgrpo_v_dataloader(
                config.data, tokenizer, processor
            )
        else:
            print("Creating standard dataloaders using 'create_dataloader'...")
            train_dataloader, val_dataloader = create_dataloader(
                config.data, tokenizer, processor
            )
        print("Dataloaders created successfully.")
        # =================================================================

        # 初始化并运行 Trainer
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    log_dir = "/home/zhq/workdir/GUI/EasyR1/verl/trainer/train_log"
    os.makedirs(log_dir, exist_ok=True) # 确保日志目录存在
    
    # 创建一个基于当前时间的、独一无二的日志文件名
    log_filename = f"mgrpo_v_run_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # 2. 配置 logging 模块
    logging.basicConfig(
        level=logging.INFO,  
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_filepath,
        filemode='w'           
    )

    # 从 CLI 和配置文件加载 PPOConfig
    cli_args       = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        path = cli_args.pop("config")
        print(f"Loading configuration from: {path}")
        file_conf = OmegaConf.load(path)
        default_config = OmegaConf.merge(default_config, file_conf)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    # 初始化 Ray
    if not ray.is_initialized():
        print("Initializing Ray...")
        envs = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
            }
        }
        ray.init(runtime_env=envs)
        print("Ray initialized.")

    # 启动远程 Runner
    runner = Runner.remote()
    print("Starting the training runner...")
    ray.get(runner.run.remote(ppo_config))
    print("Training finished.")


if __name__ == "__main__":
    main()
