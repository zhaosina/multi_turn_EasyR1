
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Dict, Literal, Tuple, List
import re

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF
import logging

if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf"""
    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller."""
    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"
    MGRPO_V = "mgrpo_v"  # New estimator for multi-turn visual grounding

def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl

def _extract_bbox_from_string(text: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # 匹配 <box>(x1, y1), (x2, y2)</box> 格式
    m = re.search(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)', text)
    if m:
        x1, y1, x2, y2 = map(int, m.groups())
        return ((x1, y1), (x2, y2))
    raise ValueError("Bounding box format not found in string.")


def _parse_absolute_coords_from_response(resp: str, img_w: int, img_h: int) -> List[int]:

    resp = resp.replace('<|im_end|>', '').strip()
    # 1. 优先匹配最完整、最明确的格式: [[x, y]]
    m = re.search(r'CLICK\s+<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    
    m = re.search(r'actions?:\s*CLICK\s+<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    
    m = re.search(r'<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    # 2. 匹配XML风格的格式，并使用 s? 来兼容 <point> 和 <points>
    m = re.search(r'<points?[^>]*x1\s*=\s*[\'"]?([\d.]+)[\'"]?[^>]*y1\s*=\s*[\'"]?([\d.]+)[\'"]?', resp, flags=re.I)
    if m:
        return [int(float(m.group(1))), int(float(m.group(2)))]

    # 3. 匹配包含浮点数的格式，并处理相对/绝对坐标转换
    m = re.search(r'<point>\[(\d+\.?\d*),\s*(\d+\.?\d*)\]</point>', resp, flags=re.I)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x <= 1.0 and y <= 1.0:
            return [int(x * img_w), int(y * img_h)]
        return [int(x), int(y)]

    # 4. 匹配通用括号格式 ( ... ) or [ ... ]
    m = re.search(r'[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\]\)]', resp)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x <= 1.0 and y <= 1.0:
            return [int(x * img_w), int(y * img_h)]
        return [int(x), int(y)]

    # 5. 如果包含 'box' 关键字，尝试解析bounding box并计算中心点作为后备方案
    if 'box' in resp.lower():
        try:
            b = _extract_bbox_from_string(resp)
            center_x = (b[0][0] + b[1][0]) / 2
            center_y = (b[0][1] + b[1][1]) / 2
            return [int(center_x), int(center_y)]
        except ValueError:
            pass
    raise ValueError("no_coord_found_in_response")

def _check_hit_absolute(pred_abs: List[int], gt_bbox_xywh: List[float]) -> bool:

    if not pred_abs or len(pred_abs) != 2: return False
    pred_x, pred_y = pred_abs
    x, y, w, h = gt_bbox_xywh
    gt_x1, gt_y1, gt_x2, gt_y2 = x, y, x + w, y + h
    return (gt_x1 <= pred_x <= gt_x2 and gt_y1 <= pred_y <= gt_y2)

def _compute_distance_absolute(pred_abs: List[int], gt_bbox_xywh: List[float]) -> float:

    if not pred_abs or len(pred_abs) != 2: return float('inf')
    pred_x, pred_y = pred_abs
    x, y, w, h = gt_bbox_xywh
    gt_center_x = x + w / 2
    gt_center_y = y + h / 2
    return np.sqrt((pred_x - gt_center_x)**2 + (pred_y - gt_center_y)**2)

def _calculate_distance(point1: List[float], point2: List[float]) -> float:
    p1 = np.array(point1)
    p2 = np.array(point2)
    return np.linalg.norm(p1 - p2)


def _score_trajectory(trajectory_coords: List[List[int]], gt_bbox_xywh: List[float], img_dims: List[int]) -> Dict[str, float]:

    if not trajectory_coords:
        # 如果轨迹无效，返回一个包含惩罚分数的字典
        return {
            "total": -10.0,
            "accuracy": 0.0,
            "efficiency": 0.0,
            "smoothness": 0.0,
        }

    gt_x, gt_y, gt_w, gt_h = gt_bbox_xywh
    target_center = [gt_x + gt_w / 2, gt_y + gt_h / 2]
    
    # 1. 准确度得分
    final_prediction = trajectory_coords[-1]
    final_distance = _calculate_distance(final_prediction, target_center)
    accuracy_score = np.exp(-final_distance / 50.0)

    # 2. 效率得分 
    efficiency_score = 1.0 / len(trajectory_coords)

    # 3. 平滑度得分
    smoothness = 0
    if len(trajectory_coords) > 2:
        for i in range(2, len(trajectory_coords)):
            step_size = _calculate_distance(trajectory_coords[i], trajectory_coords[i-1])
            prev_step_size = _calculate_distance(trajectory_coords[i-1], trajectory_coords[i-2])
            if step_size < prev_step_size:
                smoothness += 1
    smoothness_score = smoothness / max(len(trajectory_coords) - 2, 1)

    # 组合最终得分
    final_score = accuracy_score * (1 + 0.2 * efficiency_score + 0.2 * smoothness_score)
    
    return {
        "total": final_score,
        "accuracy": accuracy_score,
        "efficiency": efficiency_score,
        "smoothness": smoothness_score,
    }


def _calculate_format_reward(response: str) -> float:
    """对输出的格式进行评分"""
    reward = 0.0
    if not isinstance(response, str):
        return reward
        
    if "CLICK" in response: reward += 0.1
    if "<point>" in response: reward += 0.1
    if "</point>" in response: reward += 0.1
    if "[[" in response and "]]" in response: reward += 0.2
    return reward
    
@torch.no_grad()
def compute_mgrpo_v_advantage(
    decoded_responses: np.ndarray,
    gt_bboxes_xywh: torch.Tensor,
    image_dims: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_ids: torch.Tensor,
    turn_ids: torch.Tensor,
    n_rollouts: int,
    eps: float = 1e-8,
    w_outcome: float = 1.0,
    w_prog: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size = response_mask.shape[0]
    device = response_mask.device
    
    trajectories_indices = defaultdict(list)
    # --- 1. 按 "原始任务ID" 进行分组 ---
    for i in range(batch_size):
        original_group_id = trajectory_ids[i].item() // n_rollouts
        trajectories_indices[original_group_id].append(i)

    advantages = torch.zeros_like(response_mask, dtype=torch.float32)
    returns = torch.zeros_like(response_mask, dtype=torch.float32)
    
    log_counter = 0
    log_limit = 3  # 只打印每个批次中前3个轨迹组的详细日志

    # --- 2. 为每个轨迹组计算优势 ---
    for group_id, group_indices in trajectories_indices.items():
        
        should_log = log_counter < log_limit
        if should_log:
            logging.warning(f"\n--- [MGRPO-V DEBUG] START Group ID: {group_id} ---")

        group_scores = []
        sub_trajectories = defaultdict(list)
        for idx in group_indices:
            sub_trajectories[trajectory_ids[idx].item()].append(idx)

        # 对组内的每个完整轨迹进行评分
        for traj_id_in_group, step_indices in sub_trajectories.items():
            
            sorted_step_indices = sorted(step_indices, key=lambda i: turn_ids[i].item())
            coords_list = []
            img_w, img_h = image_dims[sorted_step_indices[0]].tolist()
            
            if should_log:
                logging.warning(f"  - Trajectory ID: {traj_id_in_group}")
            for step_idx in sorted_step_indices:
                resp = decoded_responses[step_idx]
                try:
                    coords = _parse_absolute_coords_from_response(resp, img_w, img_h)
                    coords_list.append(coords)
                    if should_log:
                        logging.warning(f"    - Turn {turn_ids[step_idx].item()}: RAW='{resp.strip()}' -> PARSED={coords}")
                except ValueError:
                    if should_log:
                        logging.warning(f"    - Turn {turn_ids[step_idx].item()}: RAW='{resp.strip()}' -> PARSING FAILED")
                    coords_list = []
                    break
            
            gt_bbox = gt_bboxes_xywh[sorted_step_indices[0]].tolist()
            score_dict = _score_trajectory(coords_list, gt_bbox, [img_w, img_h])
            final_response = decoded_responses[sorted_step_indices[-1]] if sorted_step_indices else ""
            format_reward = _calculate_format_reward(final_response)
            total_score = w_outcome * score_dict["total"] + w_prog * format_reward
            group_scores.append(total_score)
            
            if should_log:
                # [日志] 打印轨迹的最终评分
                logging.warning(f"    -> SCORES: [Total: {score_dict['total']:.3f}, Acc: {score_dict['accuracy']:.2f}, Eff: {score_dict['efficiency']:.2f}, Smooth: {score_dict['smoothness']:.2f}]")
                logging.warning(f"    -> FORMAT_REWARD: {format_reward:.2f}")
                logging.warning(f"    -> FINAL_WEIGHTED_SCORE: {total_score:.3f}")

        # GRPO优势计算
        if not group_scores or len(group_scores) <= 1: 
            continue
            
        scores_tensor = torch.tensor(group_scores, device=device)
        mean_score = scores_tensor.mean()
        std_score = scores_tensor.std(unbiased=False)

        if std_score < eps:
            traj_advantages = torch.zeros_like(scores_tensor)
        else:
            traj_advantages = (scores_tensor - mean_score) / (std_score + eps)
        
        # 广播优势
        for i, traj_id_in_group in enumerate(sub_trajectories.keys()):
            indices_for_this_traj = (trajectory_ids == traj_id_in_group)
            advantages[indices_for_this_traj] = traj_advantages[i]
        
        if should_log:
            # [日志] 打印整个组的优势计算结果
            logging.warning(f"  - Group Final Scores: {[f'{s:.3f}' for s in group_scores]}")
            logging.warning(f"  - Group Advantage Stats: Mean={mean_score:.3f}, Std={std_score:.3f}")
            logging.warning(f"  - Calculated Advantages: {[f'{adv:.3f}' for adv in traj_advantages]}")
            logging.warning(f"--- [MGRPO-V DEBUG] END Group ID: {group_id} ---\n")
            log_counter += 1
            
    advantages = advantages * response_mask
    returns = advantages.clone()

    return advantages, returns

@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns

@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        assert len(id2score[idx]) > 1, "GRPO needs rollout.n > 1."
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor(id2score[idx]))
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns

@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))
    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns

@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        running_return = running_return * response_mask[:, t]
    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns

@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns

def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio

def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")

def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_avg_mode: Literal["token", "seq"],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    negative_approx_kl = log_probs - old_log_probs
    negative_approx_kl = torch.clamp(negative_approx_kl, -20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)
    clipped_ratio = torch.exp(
        torch.clamp(negative_approx_kl, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    metrics = {"ppo_kl": -negative_approx_kl}
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)

    pg_loss = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss3 = -advantages * clip_ratio_dual

    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)
    metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)
    final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    return final_pg_loss, metrics

def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> Tuple[torch.Tensor, float]:
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    vf_clipfrac = VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item()
    return vf_loss, vf_clipfrac

def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs
    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()
    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()
    if kl_penalty == "low_var_kl":
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)
    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)
    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
