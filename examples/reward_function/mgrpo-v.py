import re
import numpy as np
from typing import Any, Dict, List


def _parse_absolute_coords_from_response(resp: str, img_w: int, img_h: int) -> List[int]:
    """从模型的文本回复中解析出绝对坐标"""
    resp = resp.replace('<|im_end|>', '').strip()
    m = re.search(r'CLICK\s*<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m: return [int(m.group(1)), int(m.group(2))]
    m = re.search(r'\[\[(\d+),\s*(\d+)\]\]', resp)
    if m: return [int(m.group(1)), int(m.group(2))]
    raise ValueError(f"无法在回复中找到坐标: '{resp}'")

def _calculate_distance(point1: List[float], point2: List[float]) -> float:
    """计算两点之间的欧式距离"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def _score_trajectory(trajectory_coords: List[List[int]], gt_bbox_xywh: List[float]) -> Dict[str, float]:
    """对一条完整的坐标轨迹进行评分"""
    if not trajectory_coords:
        return {"total": -10.0, "accuracy": 0.0, "efficiency": 0.0, "smoothness": 0.0}

    gt_x, gt_y, gt_w, gt_h = gt_bbox_xywh
    target_center = [gt_x + gt_w / 2, gt_y + gt_h / 2]
    
    final_prediction = trajectory_coords[-1]
    final_distance = _calculate_distance(final_prediction, target_center)
    accuracy_score = np.exp(-final_distance / 50.0) 

    efficiency_score = 1.0 / len(trajectory_coords)

    smoothness = 0
    if len(trajectory_coords) > 2:
        for i in range(2, len(trajectory_coords)):
            step_size = _calculate_distance(trajectory_coords[i], trajectory_coords[i-1])
            prev_step_size = _calculate_distance(trajectory_coords[i-1], trajectory_coords[i-2])
            if step_size < prev_step_size:
                smoothness += 1
    smoothness_score = smoothness / max(len(trajectory_coords) - 2, 1)

    outcome_total_score = accuracy_score * (1 + 0.2 * efficiency_score + 0.2 * smoothness_score)
    
    return {
        "total": outcome_total_score,
        "accuracy": accuracy_score,
        "efficiency": efficiency_score,
        "smoothness": smoothness_score,
    }

def _calculate_format_reward(response: str) -> float:
    """对输出的格式进行评分"""
    reward = 0.0
    if not isinstance(response, str): return reward
    if "CLICK" in response: reward += 0.1
    if "<point>" in response: reward += 0.1
    if "</point>" in response: reward += 0.1
    if "[[" in response and "]]" in response: reward += 0.2
    return reward

def compute_trajectory_score(reward_input: Dict[str, Any], w_outcome: float = 1.0, w_prog: float = 0.5) -> Dict[str, float]:
    """
    计算单条完整轨迹的综合得分。
    """

    if not isinstance(reward_input, dict):
        raise ValueError("输入必须是一个字典。")
    required_keys = ["decoded_responses", "gt_bbox_xywh", "image_dims"]
    if not all(key in reward_input for key in required_keys):
        raise ValueError(f"输入字典缺少必要的键，需要: {required_keys}")
    img_w, img_h = reward_input["image_dims"]
    coords_list = []
    for resp in reward_input["decoded_responses"]:
        try:
            coords = _parse_absolute_coords_from_response(resp, img_w, img_h)
            coords_list.append(coords)
        except ValueError:

            coords_list = []
            break
    outcome_scores = _score_trajectory(coords_list, reward_input["gt_bbox_xywh"])
    final_response = reward_input["decoded_responses"][-1] if reward_input["decoded_responses"] else ""
    format_score = _calculate_format_reward(final_response)
    
    overall_score = w_outcome * outcome_scores["total"] + w_prog * format_score
    
    return {
        "overall": overall_score,
        "outcome_total": outcome_scores["total"],
        "accuracy": outcome_scores["accuracy"],
        "efficiency": outcome_scores["efficiency"],
        "smoothness": outcome_scores["smoothness"],
        "format": format_score,
    }