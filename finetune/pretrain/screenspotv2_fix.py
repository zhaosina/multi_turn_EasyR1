import torch
import argparse
import os
import json
import logging
import re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoProcessor,
    AutoModelForVision2Seq, GenerationConfig
)
import numpy as np
from typing import List, Tuple, Dict, Any

def resize_and_transform_bbox(
    img: Image.Image, 
    bbox_xywh: List[float], 
    target_size: tuple, 
    fill_color: tuple = (128, 128, 128)
) -> Tuple[Image.Image, List[float]]:
    """
    Resizes an image to a target size while maintaining aspect ratio,
    and transforms the bounding box coordinates accordingly.

    :param img: The input PIL Image.
    :param bbox_xywh: The ground-truth bounding box in [x, y, w, h] format.
    :param target_size: A tuple (width, height) for the output image.
    :param fill_color: The color for the padding. Default is gray.
    :return: A tuple containing the resized/padded PIL Image and the transformed bbox 
             in [x', y', w', h'] format.
    """
    original_w, original_h = img.size
    target_w, target_h = target_size
    ratio = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * ratio)
    new_h = int(original_h * ratio)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", target_size, fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    original_x, original_y, original_box_w, original_box_h = bbox_xywh
    transformed_x = original_x * ratio + paste_x
    transformed_y = original_y * ratio + paste_y
    transformed_w = original_box_w * ratio
    transformed_h = original_box_h * ratio
    transformed_bbox_xywh = [transformed_x, transformed_y, transformed_w, transformed_h]
    return new_img, transformed_bbox_xywh

def get_system_prompt() -> str:
    """获取系统提示"""
    return """You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
Thoughts: Clearly outline your reasoning process for current step.
Actions: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating.

Your current task instruction, action history, and associated screenshot are as follows:
Screenshot: """

def get_task_prompts() -> List[str]:
    """获取任务提示模板"""
    return [
        "Task instruction: {}. Note: The image is in a 448x448 frame, please provide coordinates relative to this frame.\nHistory: null"
        "Task instruction: {}\nHistory: null",
        "Task instruction: {}\nHistory: Previous attempts marked with colored dots on the screenshot.",
        "Task instruction: {}\nHistory: Multiple previous attempts shown as colored markers. Consider previous predictions when making your decision."
    ]

def draw_point_on_image(image: Image.Image, x_abs: float, y_abs: float, iteration: int, color: tuple = None) -> Image.Image:
    """在图像上绘制预测点（使用绝对坐标）"""
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  
    point_color = color or colors[iteration % len(colors)]
    radius = 8
    x_pixel, y_pixel = int(x_abs), int(y_abs)
    img_w, img_h = image.size
    x_pixel = max(0, min(x_pixel, img_w - 1))
    y_pixel = max(0, min(y_pixel, img_h - 1))
    draw.ellipse([x_pixel-radius, y_pixel-radius, x_pixel+radius, y_pixel+radius], 
                 fill=point_color, outline=(255, 255, 255), width=2) 
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    draw.text((x_pixel+radius+2, y_pixel-radius), f"#{iteration+1}", 
              fill=point_color, font=font)
    return new_image

def _extract_bbox_from_string(text: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # 匹配 <box>(x1, y1), (x2, y2)</box> 格式
    m = re.search(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)', text)
    if m:
        x1, y1, x2, y2 = map(int, m.groups())
        return ((x1, y1), (x2, y2))
    raise ValueError("Bounding box format not found in string.")


def parse_qwen25vl_absolute_coords(resp: str, img_w: int, img_h: int) -> List[int]:

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

def check_hit_absolute_coords(pred_abs: List[int], bbox_gt_xyxy: List[float]) -> bool:
    """使用绝对坐标进行命中检测"""
    if not pred_abs or len(pred_abs) != 2: return False
    pred_x, pred_y = pred_abs[0], pred_abs[1]
    hit = (bbox_gt_xyxy[0] <= pred_x <= bbox_gt_xyxy[2] and 
           bbox_gt_xyxy[1] <= pred_y <= bbox_gt_xyxy[3])
    return hit

def process_iteration_with_system_prompt(model, processor, tokenizer, image, task_instruction, iteration):
    """处理单次迭代预测 (系统提示模式)"""
    sys_prompt = get_system_prompt()
    task_prompts = get_task_prompts()
    task_prompt = task_prompts[min(iteration, len(task_prompts) - 1)].format(task_instruction)
    full_prompt = sys_prompt + "\n\n" + task_prompt
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text",  "text": full_prompt}]}]
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_in], images=[image], return_tensors="pt", padding=True)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return response

def process_iteration_simple_format(model, processor, tokenizer, image, task_instruction, iteration):
    """处理单次迭代预测 (简单格式模式)"""
    if iteration == 0:
        prompt_tpl = 'In this UI screenshot, what is the position of the element corresponding to the command "{}" (with point)?'
        prompt = ('Only return "<point>[x, y]</point>".\n\n' + prompt_tpl.format(task_instruction))
    else:
        prompt_tpl = 'In this UI screenshot with previous attempts marked, what is the position of the element corresponding to the command "{}" (with point)?'
        prompt = ('Only return "<point>[x, y]</point>". Previous attempts are shown as colored dots. Learn from them.\n\n' + prompt_tpl.format(task_instruction))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_in], images=[image], return_tensors="pt", padding=True)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)
    out_ids = model.generate(**inputs, max_new_tokens=64)
    response = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen_path', type=str, required=True)         
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--log_file', type=str, default='eval_fixed_res.log')
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--save_iteration_images', action='store_true')
    parser.add_argument('--use_system_prompt', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(args.log_file, mode='w', encoding='utf-8'),
                                  logging.StreamHandler()])
    torch.manual_seed(1234)
    
    logging.info("Loading model and processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.qwen_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(args.qwen_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16).eval()
    model.generation_config = GenerationConfig.from_pretrained(args.qwen_path, trust_remote_code=True)
    logging.info("Model and processor loaded.")

    tasks = (["desktop", "mobile", "web"] if args.task == "all" else [args.task])
    debug_dir = os.path.join(args.screenspot_test, "debug_logs_fixed_res")
    iteration_imgs_dir = os.path.join(debug_dir, "iteration_images")
    os.makedirs(debug_dir, exist_ok=True)
    if args.save_iteration_images:
        os.makedirs(iteration_imgs_dir, exist_ok=True)

    for task in tasks:
        json_path = os.path.join(args.screenspot_test, f"screenspot_{task}_v2.json")
        if not os.path.exists(json_path):
            logging.error(f"Annotation file not found: {json_path}")
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"--- Starting Subtask `{task}`: {len(data)} samples, using {args.num_iterations} iterations ---")
        logging.info(f"Prompt mode: {'System Prompt' if args.use_system_prompt else 'Simple Format'}")

        all_outputs, wrong_format, no_hit = [], [], []
        num_action, corr_action = 0, 0
        original_model_correct, original_model_total = 0, 0
        iteration_stats = {i: {"correct": 0, "total": 0} for i in range(args.num_iterations)}

        for idx, item in tqdm(list(enumerate(data)), desc=f"Processing {task}"):
            img_file = os.path.join(args.screenspot_imgs, item["img_filename"])
            if not os.path.exists(img_file):
                logging.warning(f"[{task}][{idx}] Missing image {img_file}")
                wrong_format.append({"idx": idx, "reason": "img_missing"})
                continue

            # --- 统一处理图像和坐标 ---
            target_size = (448, 448)
            original_image = Image.open(img_file).convert("RGB")
            original_bbox_xywh = item["bbox"]

            padded_image, transformed_bbox_xywh = resize_and_transform_bbox(
                original_image, original_bbox_xywh, target_size
            )
            
            tx, ty, tw, th = transformed_bbox_xywh
            bbox_gt_transformed_xyxy = [tx, ty, tx + tw, ty + th]
            
            current_image = padded_image.copy()
            instr = item["instruction"]
            # --- 修改结束 ---
            
            iteration_results = []
            final_hit = False
            first_iteration_hit = False

            for iteration in range(args.num_iterations):
                if final_hit:
                    break
                    
                try:
                    if args.use_system_prompt:
                        response = process_iteration_with_system_prompt(model, processor, tokenizer, current_image, instr, iteration)
                    else:
                        response = process_iteration_simple_format(model, processor, tokenizer, current_image, instr, iteration)
                     # 修改为记录完整、清晰的原始输出
                    logging.info(
                        f"[{task}][{idx}] Iteration {iteration + 1} Raw Model Output:\n"
                        f"-------------------- START RESPONSE --------------------\n"
                        f"{response.strip()}\n"
                        f"--------------------  END RESPONSE  --------------------"
                    )
                    pred_abs = parse_qwen25vl_absolute_coords(response, target_size[0], target_size[1])
                    hit = check_hit_absolute_coords(pred_abs, bbox_gt_transformed_xyxy)

                    if iteration == 0:
                        original_model_total += 1
                        if hit:
                            first_iteration_hit = True
                            original_model_correct += 1
                            logging.info(f"[{task}][{idx}] 第一次迭代命中! 坐标: {pred_abs}")
                        else:
                            logging.info(f"[{task}][{idx}] 第一次迭代未命中, 坐标: {pred_abs}")
                    
                    iteration_results.append({"iteration": iteration, "response": response, "prediction_abs": pred_abs, "hit": hit})
                    
                    current_image = draw_point_on_image(current_image, pred_abs[0], pred_abs[1], iteration)
                    
                    if args.save_iteration_images:
                        iter_img_path = os.path.join(iteration_imgs_dir, f"{task}_{idx}_iter_{iteration+1}.png")
                        current_image.save(iter_img_path)
                    
                    iteration_stats[iteration]["total"] += 1
                    if hit:
                        iteration_stats[iteration]["correct"] += 1
                        final_hit = True
                        logging.info(f"[{task}][{idx}] 迭代 {iteration+1}: 命中! 停止后续迭代")
                    else:
                        logging.info(f"[{task}][{idx}] 迭代 {iteration+1}: 未命中，绝对坐标: {pred_abs}")
                        if iteration == args.num_iterations - 1:
                            logging.info(f"[{task}][{idx}] 所有 {args.num_iterations} 次迭代均未命中")
                    
                except Exception as e:
                    logging.error(f"[{task}][{idx}] Iteration {iteration+1} Error: {e}")
                    iteration_results.append({"iteration": iteration, "error": str(e), "hit": False})
                    if iteration == 0:
                        original_model_total += 1
                        logging.info(f"[{task}][{idx}] 第一次迭代错误")
            
            num_action += 1
            if final_hit:
                corr_action += 1
            else:
                 no_hit.append({"idx": idx, "item": item})
            
            all_outputs.append({
                "idx": idx,
                "instruction": instr,
                "original_bbox_xywh": original_bbox_xywh,
                "transformed_bbox_xywh": transformed_bbox_xywh,
                "iteration_results": iteration_results,
                "final_hit": final_hit,
            })

        # --- 结果保存与统计 ---
        suffix = "_sys" if args.use_system_prompt else "_simple"
        with open(os.path.join(debug_dir, f"all_outputs_{task}{suffix}.json"), "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
        with open(os.path.join(debug_dir, f"no_hit_{task}{suffix}.json"), "w", encoding="utf-8") as f:
            json.dump(no_hit, f, ensure_ascii=False, indent=2)

        original_model_acc = original_model_correct / max(original_model_total, 1)
        multi_iteration_acc = corr_action / max(num_action, 1)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"TASK {task} FINAL STATISTICS (FIXED RESOLUTION):")
        logging.info(f"{'='*60}")
        logging.info(f"Original Model Accuracy (1st iter): {original_model_acc:.3f} ({original_model_correct}/{original_model_total})")
        logging.info(f"Multi-iteration Accuracy (any iter): {multi_iteration_acc:.3f} ({corr_action}/{num_action})")
        logging.info(f"Accuracy Improvement: {multi_iteration_acc - original_model_acc:.3f}")
        
        summary_stats = {
            "task": task, "prompt_mode": "system_prompt" if args.use_system_prompt else "simple_format",
            "original_model_accuracy": original_model_acc,
            "multi_iteration_accuracy": multi_iteration_acc,
            "iteration_stats": {f"iter_{i+1}": (stats['correct']/stats['total'] if stats['total']>0 else 0) for i, stats in iteration_stats.items()}
        }
        with open(os.path.join(debug_dir, f"summary_stats_{task}{suffix}.json"), "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    logging.info("All tasks completed!")

if __name__ == "__main__":
    main()