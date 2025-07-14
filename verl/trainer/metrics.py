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

from typing import Any, Dict, List

import numpy as np
import torch

from ..protocol import DataProto
from ..utils.py_functional import convert_dict_to_str

def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> Dict[str, Any]:
    """Compute metrics related to the data in the batch."""
    metrics = {}

    if "response_mask" not in batch.batch:
        return {}

    response_mask = batch.batch["response_mask"].bool()

    # Path for MGRPO-V where advantages are computed at the trajectory level
    if "token_level_rewards" not in batch.batch and "advantages" in batch.batch:
        valid_adv = batch.batch["advantages"][response_mask]
        metrics.update({
            "critic/advantages_mean": valid_adv.mean().detach().item(),
            "critic/advantages_max": valid_adv.max().detach().item(),
            "critic/advantages_min": valid_adv.min().detach().item(),
        })
        return metrics

    if "token_level_scores" not in batch.batch:
        return {} # Not enough data for full metrics

    token_level_scores = batch.batch["token_level_scores"]
    token_level_rewards = batch.batch["token_level_rewards"]
    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    # Sequence-level metrics
    metrics.update({
        "critic/score_mean": (token_level_scores * response_mask).sum(dim=-1).mean().detach().item(),
        "critic/reward_mean": (token_level_rewards * response_mask).sum(dim=-1).mean().detach().item(),
    })

    # Token-level metrics for advantage and return
    valid_adv = advantages[response_mask]
    valid_returns = returns[response_mask]
    metrics.update({
        "critic/advantages_mean": valid_adv.mean().detach().item(),
        "critic/advantages_max": valid_adv.max().detach().item(),
        "critic/advantages_min": valid_adv.min().detach().item(),
        "critic/returns_mean": valid_returns.mean().detach().item(),
        "critic/returns_max": valid_returns.max().detach().item(),
        "critic/returns_min": valid_returns.min().detach().item(),
    })

    # Critic/Value-specific metrics
    if use_critic and "values" in batch.batch:
        values = batch.batch["values"]
        valid_values = values[response_mask]
        
        # Explained variance
        explained_var = 1 - torch.var(valid_returns - valid_values) / (torch.var(valid_returns) + 1e-8)

        metrics.update({
            "critic/values_mean": valid_values.mean().detach().item(),
            "critic/values_max": valid_values.max().detach().item(),
            "critic/values_min": valid_values.min().detach().item(),
            "critic/vf_explained_var": explained_var.detach().item(),
        })
    
    # Length metrics
    if "attention_mask" in batch.batch:
        attention_mask = batch.batch["attention_mask"].bool()
        response_length = response_mask.sum(dim=-1).float()
        # Prompt length is total length minus response length
        prompt_length = attention_mask.sum(dim=-1).float() - response_length
        
        metrics.update({
            "response_length/mean": response_length.mean().detach().item(),
            "response_length/max": response_length.max().detach().item(),
            "response_length/min": response_length.min().detach().item(),
            "prompt_length/mean": prompt_length.mean().detach().item(),
            "prompt_length/max": prompt_length.max().detach().item(),
            "prompt_length/min": prompt_length.min().detach().item(),
        })

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    if "response_mask" not in batch.batch or "global_token_num" not in batch.meta_info:
        return {} # Not enough info to compute timing

    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], num_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }