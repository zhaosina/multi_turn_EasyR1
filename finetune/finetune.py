# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    qwen_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"]  ##["in_proj","out_proj","c_fc"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                          _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    # print(torch.ne(input_ids, 151643).sum().item())

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
    )

    # customized LoRA parameters
    target_modules = []
    target_layer_names = ["visual.conv1", "attn.in_proj", "attn.out_proj", "mlp.c_fc", "mlp.c_proj", "c_attn",
                          "attn.c_proj", "w1", "w2"]
    lora_supported_types = [torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, transformers.pytorch_utils.Conv1D]
    for name, module in model.named_modules():
        if any(t_name in name for t_name in target_layer_names) and 'attn_pool' not in name:
            if isinstance(module, tuple(lora_supported_types)):
                target_modules.append(name)
            else:
                print(name + " not satisfy lora")
                input()
    lora_args.lora_target_modules = target_modules

    """
    # print the LoRA parameters
    for name, param in model.named_parameters():
        if any(target in name for target in lora_args.lora_target_modules):
            print(name)
    """

    if not training_args.use_lora:
        if training_args.fix_vit and hasattr(model, 'transformer') and hasattr(model.transformer, 'visual'):
            model.transformer.visual.requires_grad_(False)
            if hasattr(model.transformer.visual, 'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.qwen_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eod_id

    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()


@torch.no_grad()
def compute_mgrpo_v_advantage(
    decoded_responses: List[str],
    gt_bboxes_xywh: torch.Tensor,
    image_dims: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_ids: torch.Tensor,
    turn_ids: torch.Tensor,
    w_outcome: float = 1.0,
    w_prog: float = 0.5,
    alpha_dist: float = 0.001, # 距离惩罚的权重需要根据绝对坐标系重新调整
    beta_impr: float = 0.1,    # 改进奖励的权重也需要重新调整
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes advantage for a multi-turn visual grounding task (MGRPO-V) with dense process rewards.
    This function calculates rewards based on the entire trajectory, incorporating:
    1.  Outcome Reward (R_out): Whether the final prediction was a hit.
    2.  Process Reward (R_prog): A combination of distance decay and relative improvement.
    The final rewards are Z-score normalized and clipped to enhance training stability.

    Args:
        decoded_responses (List[str]): Decoded text responses for the entire batch.
        gt_bboxes_xywh (torch.Tensor): Ground truth bboxes in [x,y,w,h] format. Shape: (num_trajectories, 4).
        image_dims (torch.Tensor): Image dimensions (width, height) for each trajectory. Shape: (num_trajectories, 2).
        response_mask (torch.Tensor): Response mask for the batch. Shape: (batch_size, response_length).
        trajectory_ids (torch.Tensor): Groups sequences into trajectories. Shape: (batch_size,).
        turn_ids (torch.Tensor): The turn number for each sequence. Shape: (batch_size,).
        w_outcome (float): Weight for the outcome reward.
        w_prog (float): Weight for the process reward.
        alpha_dist (float): Scaling factor for the distance decay reward.
        beta_impr (float): Scaling factor for the improvement reward.
        eps (float): Epsilon for stable normalization.

    Returns:
        advantages (torch.Tensor): The computed advantages.
        returns (torch.Tensor): The returns (advantages + values). Here, they are the same.
    """
    batch_size = response_mask.shape[0]
    device = response_mask.device
    scalar_rewards = torch.zeros(batch_size, device=device)
    trajectories = defaultdict(list)
    for i in range(batch_size):
        trajectories[trajectory_ids[i].item()].append(i)

    for traj_id, indices in trajectories.items():
        initial_turn_idx, final_turn_idx, max_turn = -1, -1, -1
        for i in indices:
            if turn_ids[i].item() == 0: initial_turn_idx = i
            if turn_ids[i].item() > max_turn:
                max_turn = turn_ids[i].item()
                final_turn_idx = i
        
        if initial_turn_idx == -1 or final_turn_idx == -1: continue

        first_sample_idx_in_batch = indices[0]
        gt_bbox = gt_bboxes_xywh[first_sample_idx_in_batch].tolist()
        img_w, img_h = image_dims[first_sample_idx_in_batch].tolist()

        try:
            initial_coords = _parse_absolute_coords_from_response(decoded_responses[initial_turn_idx], img_w, img_h)
            final_coords = _parse_absolute_coords_from_response(decoded_responses[final_turn_idx], img_w, img_h)
            
            final_hit = _check_hit_absolute(final_coords, gt_bbox)
            dist_initial = _compute_distance_absolute(initial_coords, gt_bbox)
            dist_final = _compute_distance_absolute(final_coords, gt_bbox)
            initial_hit = _check_hit_absolute(initial_coords, gt_bbox)

        except Exception:
            final_hit, initial_hit = False, False
            dist_initial, dist_final = float('inf'), float('inf')

        # --- 基于绝对坐标重新设计奖励 ---
        R_outcome = 1.0 if final_hit else -0.1 # 命中得1分，未命中得-0.1分

        r_dist_decay = -alpha_dist * dist_final
        #改进奖励，若第二次预测比第一次更接近目标，则给予奖励
        r_improvement = 0.0
        if max_turn > 0 and dist_initial != float('inf'):
            r_improvement = beta_impr * (dist_initial - dist_final)
            
        R_prog = r_dist_decay + r_improvement
        
        R_total = w_outcome * R_outcome + w_prog * R_prog

        for i in indices:
            scalar_rewards[i] = R_total
    
    advantages = scalar_rewards.unsqueeze(-1) * response_mask
    returns = advantages.clone()

    return advantages, returns
