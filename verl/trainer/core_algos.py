
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Dict, Literal, Tuple, List
import re

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import torch_functional as VF


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


def _parse_coords_from_response(resp: str) -> List[float]:
    """
    Parses relative coordinates from the model's response string.
    """
    resp = resp.replace('<|im_end|>', '').strip()
    
    # Pattern: CLICK <point>[[x, y]]</point>
    m = re.search(r'CLICK\s+<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        return [int(m.group(1)) / 1000.0, int(m.group(2)) / 1000.0]
    
    # Pattern: actions: CLICK <point>[[x, y]]</point>
    m = re.search(r'actions?:\s*CLICK\s+<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        return [int(m.group(1)) / 1000.0, int(m.group(2)) / 1000.0]
    
    # Pattern: <point>[[x, y]]</point>
    m = re.search(r'<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        return [int(m.group(1)) / 1000.0, int(m.group(2)) / 1000.0]
    return []

def _convert_and_check_hit(pred_rel: List[float], gt_bbox_xywh: List[float], img_w: int, img_h: int) -> bool:
    """
    Converts ground truth bbox and checks if the prediction point hits it.
    """
    if not pred_rel or len(pred_rel) != 2:
        return False

    # 1. Convert gt bbox from [x, y, w, h] to [x1, y1, x2, y2]
    x, y, w, h = gt_bbox_xywh
    gt_bbox_xyxy = [x, y, x + w, y + h]

    # 2. Convert gt bbox to relative coordinates
    x1, y1, x2, y2 = gt_bbox_xyxy
    gt_bbox_rel = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]

    # 3. Check for hit
    px_rel, py_rel = pred_rel
    hit = (gt_bbox_rel[0] <= px_rel <= gt_bbox_rel[2] and 
           gt_bbox_rel[1] <= py_rel <= gt_bbox_rel[3])
    return hit
"""
@torch.no_grad()
def compute_mgrpo_v_advantage(
    decoded_responses: List[str],
    gt_bboxes_xywh: torch.Tensor,
    image_dims: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_ids: torch.Tensor,
    turn_ids: torch.Tensor,
    # --- Config parameters for the reward function ---
    process_reward_bonus: float = 0.5,
    process_reward_penalty: float = -0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size = response_mask.shape[0]
    device = response_mask.device
    
    scalar_rewards = torch.zeros(batch_size, device=device)

    trajectories = defaultdict(list)
    for i in range(batch_size):
        trajectories[trajectory_ids[i].item()].append(i)

    for traj_id, indices in trajectories.items():
        initial_turn_idx, final_turn_idx, max_turn = -1, -1, -1
        for i in indices:
            if turn_ids[i].item() == 0:
                initial_turn_idx = i
            if turn_ids[i].item() > max_turn:
                max_turn = turn_ids[i].item()
                final_turn_idx = i
        
        if initial_turn_idx == -1 or final_turn_idx == -1:
            continue

        gt_bbox = gt_bboxes_xywh[traj_id].tolist()
        img_w, img_h = image_dims[traj_id].tolist()

        try:
            initial_coords = _parse_coords_from_response(decoded_responses[initial_turn_idx])
            final_coords = _parse_coords_from_response(decoded_responses[final_turn_idx])

            initial_hit = _convert_and_check_hit(initial_coords, gt_bbox, img_w, img_h)
            final_hit = _convert_and_check_hit(final_coords, gt_bbox, img_w, img_h)
        except Exception:
            initial_hit, final_hit = False, False

        # 1. Outcome Reward: 1.0 for a final hit, 0.0 otherwise.
        R_outcome = 1.0 if final_hit else 0.0

        # 2. Process Reward: Bonus for successful correction.
        R_process = 0.0
        # A successful correction is when the initial attempt fails but the final one succeeds.
        if not initial_hit and final_hit:
            R_process = process_reward_bonus
        # A failed correction is when the initial attempt fails and the final one also fails.
        elif not initial_hit and not final_hit:
            R_process = process_reward_penalty
        
        R_total = R_outcome + R_process

        for i in indices:
            scalar_rewards[i] = R_total
    
    advantages = scalar_rewards.unsqueeze(-1) * response_mask
    returns = advantages.clone()

    return advantages, returns
"""

def _compute_distance_from_center(pred_rel: List[float], gt_bbox_xywh: List[float], img_w: int, img_h: int) -> float:
    """
    Computes the Euclidean distance between a predicted point and the center of the ground truth bbox.
    All calculations are done in the relative coordinate space.
    """
    if not pred_rel or len(pred_rel) != 2:
        return float('inf') # Return a large distance if prediction is invalid

    # Center of the ground truth bbox in relative coordinates
    gt_x, gt_y, gt_w, gt_h = gt_bbox_xywh
    gt_center_x_rel = (gt_x + gt_w / 2) / img_w
    gt_center_y_rel = (gt_y + gt_h / 2) / img_h
    
    pred_x_rel, pred_y_rel = pred_rel
    
    distance = np.sqrt((pred_x_rel - gt_center_x_rel)**2 + (pred_y_rel - gt_center_y_rel)**2)
    return distance

@torch.no_grad()
def compute_mgrpo_v_advantage(
    decoded_responses: List[str],
    gt_bboxes_xywh: torch.Tensor,
    image_dims: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_ids: torch.Tensor,
    turn_ids: torch.Tensor,
    # --- Config parameters for the reward function ---
    w_outcome: float = 1.0,
    w_prog: float = 0.5,
    alpha_dist: float = 1.0,
    beta_impr: float = 2.0,
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
            if turn_ids[i].item() == 0:
                initial_turn_idx = i
            if turn_ids[i].item() > max_turn:
                max_turn = turn_ids[i].item()
                final_turn_idx = i
        
        if initial_turn_idx == -1 or final_turn_idx == -1:
            continue

        gt_bbox = gt_bboxes_xywh[traj_id].tolist()
        img_w, img_h = image_dims[traj_id].tolist()

        try:
            initial_coords = _parse_coords_from_response(decoded_responses[initial_turn_idx])
            final_coords = _parse_coords_from_response(decoded_responses[final_turn_idx])

            initial_hit = _convert_and_check_hit(initial_coords, gt_bbox, img_w, img_h)
            final_hit = _convert_and_check_hit(final_coords, gt_bbox, img_w, img_h)

            dist_initial = _compute_distance_from_center(initial_coords, gt_bbox, img_w, img_h)
            dist_final = _compute_distance_from_center(final_coords, gt_bbox, img_w, img_h)

        except Exception:
            initial_hit, final_hit = False, False
            dist_initial, dist_final = float('inf'), float('inf')

        # --- Compute Multi-Objective Rewards ---

        # 1. Outcome Reward (R_out): 1.0 for a final hit, 0.0 otherwise.
        R_outcome = 1.0 if final_hit else 0.0

        # 2. Process Reward (R_prog): Dense reward based on progress.
        # 2a. Distance Decay Reward: Penalizes distance from the target center.
        r_dist_decay = -alpha_dist * dist_final
        
        # 2b. Improvement Reward: Rewards moving closer to the target. Only for multi-turn.
        r_improvement = 0.0
        if max_turn > 0 and dist_initial != float('inf'): # Check if it's a correction turn
            r_improvement = beta_impr * (dist_initial - dist_final)
            
        R_prog = r_dist_decay + r_improvement

        # 3. Semantic Consistency Reward (R_vlm)
        # R_vlm = compute_vlm_score(crop(final_coords), instruction)
        # R_total = w_outcome * R_outcome + w_prog * R_prog + w_vlm * R_vlm
        
        R_total = w_outcome * R_outcome + w_prog * R_prog

        for i in indices:
            scalar_rewards[i] = R_total
    
    # --- Reward Normalization and Shaping ---
    # Use Z-score normalization across the entire batch to reduce variance
    mean_reward = torch.mean(scalar_rewards)
    std_reward = torch.std(scalar_rewards)
    normalized_rewards = (scalar_rewards - mean_reward) / (std_reward + eps)
    
    # Use tanh to clip rewards to [-1, 1]
    shaped_rewards = torch.tanh(normalized_rewards)
    advantages = shaped_rewards.unsqueeze(-1) * response_mask
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