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
from process_utils import extract_bbox
import numpy as np

def get_system_prompt():
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

def get_task_prompts():
    """获取任务提示模板"""
    return [
        "Task instruction: {}\nHistory: null",
        "Task instruction: {}\nHistory: Previous attempts marked with colored dots on the screenshot.",
        "Task instruction: {}\nHistory: Multiple previous attempts shown as colored markers. Consider previous predictions when making your decision."
    ]

def draw_point_on_image(image, x_abs, y_abs, iteration, color=None):
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
    except:
        font = ImageFont.load_default()
    draw.text((x_pixel+radius+2, y_pixel-radius), f"#{iteration+1}", 
              fill=point_color, font=font)
    return new_image

def parse_qwen25vl_absolute_coords(resp: str, img_w: int, img_h: int):
    resp = resp.replace('<|im_end|>', '').strip()

    m = re.search(r'CLICK\s+<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return [x, y]  
    
    m = re.search(r'actions?:\s*CLICK\s+<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return [x, y]
    
    # 不带CLICK前缀的格式
    m = re.search(r'<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return [x, y]
    
    m = re.search(r'<points[^>]*x1\s*=\s*[\'"]?([\d.]+)[\'"]?[^>]*'
                  r'y1\s*=\s*[\'"]?([\d.]+)[\'"]?', resp, flags=re.I)
    if m:
        return [int(float(m.group(1))), int(float(m.group(2)))] 

    m = re.search(r'<point>\[(\d+\.?\d*),\s*(\d+\.?\d*)\]</point>', resp, flags=re.I)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        # 如果是相对坐标（0-1）转换为绝对坐标
        if x <= 1.0 and y <= 1.0:
            return [int(x * img_w), int(y * img_h)]
        return [int(x), int(y)]

    m = re.search(r'[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\]\)]', resp)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        if x <= 1.0 and y <= 1.0:
            return [int(x * img_w), int(y * img_h)]  
        return [int(x), int(y)]  # 已经是绝对坐标

    if 'box' in resp.lower():
        try:
            b = extract_bbox(resp)
            center_x = (b[0][0] + b[1][0]) / 2
            center_y = (b[0][1] + b[1][1]) / 2
            return [int(center_x), int(center_y)]
        except:
            pass
    
    raise ValueError("no_coord")

def check_hit_absolute_coords(pred_abs, bbox_gt):
    """使用绝对坐标进行命中检测"""
    pred_x, pred_y = pred_abs[0], pred_abs[1]
    
    # bbox_gt格式: [x1, y1, x2, y2] (绝对坐标)
    hit = (bbox_gt[0] <= pred_x <= bbox_gt[2] and 
           bbox_gt[1] <= pred_y <= bbox_gt[3])
    return hit

def process_iteration_with_system_prompt(model, processor, tokenizer, image, task_instruction, iteration):
    """处理单次迭代预测"""
    sys_prompt = get_system_prompt()
    task_prompts = get_task_prompts()
    task_prompt = task_prompts[min(iteration, len(task_prompts) - 1)].format(task_instruction)
    
    # 使用参考代码的消息格式，但加入系统提示
    full_prompt = sys_prompt + "\n\n" + task_prompt
    
    messages = [{"role": "user",
                 "content": [{"type": "image", "image": image},
                            {"type": "text",  "text": full_prompt}]}]

    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_in], images=[image], return_tensors="pt", padding=True)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    out_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    
    return response

def process_iteration_simple_format(model, processor, tokenizer, image, task_instruction, iteration):
    """处理单次迭代预测用简单格式"""
    if iteration == 0:
        prompt_tpl = 'In this UI screenshot, what is the position of the element corresponding to the command "{}" (with point)?'
        prompt = ('Only return "<point>[x, y]</point>".\n\n' + prompt_tpl.format(task_instruction))
    else:
        prompt_tpl = 'In this UI screenshot with previous attempts marked, what is the position of the element corresponding to the command "{}" (with point)?'
        prompt = ('Only return "<point>[x, y]</point>". Previous attempts are shown as colored dots. Learn from them.\n\n' + prompt_tpl.format(task_instruction))
    
    messages = [{"role": "user",
                 "content": [{"type": "image", "image": image},
                            {"type": "text",  "text": prompt}]}]

    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_in], images=[image], return_tensors="pt", padding=True)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    out_ids = model.generate(**inputs, max_new_tokens=64)
    response = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    
    return response

parser = argparse.ArgumentParser()
parser.add_argument('--qwen_path', type=str, required=True)         
parser.add_argument('--screenspot_imgs', type=str, required=True)
parser.add_argument('--screenspot_test', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--log_file', type=str, default='eval.log')
parser.add_argument('--num_iterations', type=int, default=3, 
                   help='Number of iterations for multi-shot prediction')
parser.add_argument('--save_iteration_images', action='store_true',
                   help='Save images with marked points for each iteration')
parser.add_argument('--use_system_prompt', action='store_true',
                   help='Use OS-Atlas style system prompt instead of simple format')
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(args.log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(args.qwen_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    args.qwen_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()
model.generation_config = GenerationConfig.from_pretrained(args.qwen_path, trust_remote_code=True)
tasks = (["desktop", "mobile", "web"] if args.task == "all" else [args.task])

debug_dir = os.path.join(args.screenspot_test, "debug_logs")
iteration_imgs_dir = os.path.join(debug_dir, "iteration_images")
os.makedirs(debug_dir, exist_ok=True)
if args.save_iteration_images:
    os.makedirs(iteration_imgs_dir, exist_ok=True)

for task in tasks:
    json_path = os.path.join(args.screenspot_test, f"screenspot_{task}_v2.json")
    if not os.path.exists(json_path):
        logging.error(f"标注文件不存在：{json_path}")
        continue

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logging.info(f"子任务 `{task}` 共 {len(data)} 条，使用 {args.num_iterations} 次迭代")
    logging.info(f"提示模式: {'系统提示' if args.use_system_prompt else '简单格式'}")

    # 统计变量
    all_outputs = []
    wrong_format = []
    no_hit = []
    num_action = 0
    corr_action = 0
    
    # 原始模型准确率统计（只有第一次迭代成功才算对）
    original_model_correct = 0
    original_model_total = 0
    
    iteration_stats = {i: {"correct": 0, "total": 0} for i in range(args.num_iterations)}

    for idx, item in tqdm(list(enumerate(data)), desc=f"{task}"):
        img_file = os.path.join(args.screenspot_imgs, item["img_filename"])
        if not os.path.exists(img_file):
            logging.warning(f"[{task}][{idx}] 缺失图片 {img_file}")
            wrong_format.append({"idx": idx, "reason": "img_missing"})
            continue

        x, y, w, h = item["bbox"]
        bbox_gt = [x, y, x + w, y + h]  # 转换为[x1, y1, x2, y2]格式
        original_image = Image.open(img_file).convert("RGB")
        img_w, img_h = original_image.size
        instr = item["instruction"]

        logging.info(f"[{task}][{idx}] GT bbox: {bbox_gt}, 图像尺寸: {img_w}x{img_h}")

        # 迭代预测
        iteration_results = []
        current_image = original_image.copy()
        final_hit = False
        first_iteration_hit = False

        for iteration in range(args.num_iterations):
            if final_hit:
                logging.info(f"[{task}][{idx}] 已在迭代 {iteration} 前命中，跳过后续迭代")
                break
                
            try:
                # 选择处理函数
                if args.use_system_prompt:
                    response = process_iteration_with_system_prompt(
                        model, processor, tokenizer, current_image, instr, iteration
                    )
                else:
                    response = process_iteration_simple_format(
                        model, processor, tokenizer, current_image, instr, iteration
                    )
                
                logging.info(f"[{task}][{idx}] 迭代 {iteration + 1} 响应: {response[:200]}...")
                
                # 解析绝对坐标
                pred_abs = parse_qwen25vl_absolute_coords(response, img_w, img_h)
                hit = check_hit_absolute_coords(pred_abs, bbox_gt)
                
                # 记录第一次迭代结果
                if iteration == 0:
                    original_model_total += 1
                    if hit:
                        first_iteration_hit = True
                        original_model_correct += 1
                        logging.info(f"[{task}][{idx}] 第一次迭代命中! 坐标: {pred_abs}")
                    else:
                        logging.info(f"[{task}][{idx}] 第一次迭代未命中, 坐标: {pred_abs}")
                
                iteration_results.append({
                    "iteration": iteration,
                    "response": response,
                    "prediction_abs": pred_abs,
                    "hit": hit,
                    "is_first_iteration": iteration == 0,
                    "first_iteration_success": first_iteration_hit if iteration == 0 else None
                })
                
                # 在图像上标记预测点（使用绝对坐标）
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
                logging.error(f"[{task}][{idx}] 迭代 {iteration+1} 错误: {e}")
                iteration_results.append({
                    "iteration": iteration,
                    "error": str(e),
                    "response": response if 'response' in locals() else "",
                    "hit": False,
                    "is_first_iteration": iteration == 0,
                    "first_iteration_success": False if iteration == 0 else None
                })
                
                if iteration == 0:
                    original_model_total += 1
                    logging.info(f"[{task}][{idx}] 第一次迭代错误")

        # 最终结果统计
        try:
            num_action += 1
            valid_predictions = [r["prediction_abs"] for r in iteration_results if "prediction_abs" in r]
            
            if valid_predictions:
                final_pred = valid_predictions[-1]
                any_hit = any(r.get("hit", False) for r in iteration_results)
                
                if any_hit:
                    corr_action += 1
                else:
                    gt_center_x = (bbox_gt[0] + bbox_gt[2]) / 2
                    gt_center_y = (bbox_gt[1] + bbox_gt[3]) / 2
                    no_hit.append({
                        "idx": idx,
                        "final_pred_abs": final_pred,
                        "all_predictions_abs": valid_predictions,
                        "gt_center": [gt_center_x, gt_center_y],
                        "bbox_gt": bbox_gt,
                        "img_size": [img_w, img_h],
                        "iterations_attempted": len(iteration_results),
                        "first_iteration_success": first_iteration_hit
                    })
                
                original_acc = original_model_correct / max(original_model_total, 1)
                multi_iter_acc = corr_action / num_action
                
                logging.info(f"[{task}][{idx}] 最终结果: {'命中' if any_hit else '未命中'}, "
                           f"原始准确率: {original_acc:.3f}, 多迭代准确率: {multi_iter_acc:.3f}")
            else:
                wrong_format.append({
                    "idx": idx,
                    "reason": "no_valid_predictions",
                    "iteration_results": iteration_results
                })
            
            all_outputs.append({
                "idx": idx,
                "instruction": instr,
                "bbox_gt": bbox_gt,
                "img_size": [img_w, img_h],
                "iteration_results": iteration_results,
                "final_prediction_abs": final_pred if valid_predictions else None,
                "final_hit": any_hit if valid_predictions else False,
                "iterations_used": len(iteration_results),
                "first_iteration_success": first_iteration_hit
            })
            
        except Exception as e:
            wrong_format.append({
                "idx": idx,
                "error": str(e),
                "iteration_results": iteration_results
            })
            logging.info(f"[{task}][{idx}] 整体错误 ({e})")

    # 保存结果
    suffix = "_multishot_syscl" if args.use_system_prompt else "_multishot_simple"
    
    with open(os.path.join(debug_dir, f"all_outputs_{task}{suffix}.json"), 
              "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(debug_dir, f"wrong_format_{task}{suffix}.json"), 
              "w", encoding="utf-8") as f:
        json.dump(wrong_format, f, ensure_ascii=False, indent=2)
        
    with open(os.path.join(debug_dir, f"no_hit_{task}{suffix}.json"), 
              "w", encoding="utf-8") as f:
        json.dump(no_hit, f, ensure_ascii=False, indent=2)

    # 最终统计
    original_model_acc = original_model_correct / max(original_model_total, 1)
    multi_iteration_acc = corr_action / max(num_action, 1)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"{task} 最终统计结果 (Qwen2.5-VL 绝对坐标):")
    logging.info(f"{'='*60}")
    logging.info(f"原始模型准确率 (仅第1次迭代): {original_model_acc:.3f} ({original_model_correct}/{original_model_total})")
    logging.info(f"多迭代总准确率 (任意迭代成功): {multi_iteration_acc:.3f} ({corr_action}/{num_action})")
    logging.info(f"准确率提升: {multi_iteration_acc - original_model_acc:.3f}")
    logging.info(f"格式错误: {len(wrong_format)}, 最终未命中: {len(no_hit)}")
    logging.info(f"提示模式: {'系统提示' if args.use_system_prompt else '简单格式'}")
    logging.info(f"\n各迭代详细统计:")
    for i in range(args.num_iterations):
        stats = iteration_stats[i]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            logging.info(f"迭代 {i+1}: {acc:.3f} ({stats['correct']}/{stats['total']})")

    # 保存汇总统计
    summary_stats = {
        "task": task,
        "coordinate_system": "absolute",
        "model_type": "Qwen2.5-VL",
        "prompt_mode": "system_prompt" if args.use_system_prompt else "simple_format",
        "total_samples": len(data),
        "original_model_accuracy": original_model_acc,
        "original_model_correct": original_model_correct,
        "original_model_total": original_model_total,
        "multi_iteration_accuracy": multi_iteration_acc,
        "multi_iteration_correct": corr_action,
        "multi_iteration_total": num_action,
        "accuracy_improvement": multi_iteration_acc - original_model_acc,
        "wrong_format_count": len(wrong_format),
        "final_no_hit_count": len(no_hit),
        "iteration_stats": iteration_stats,
        "num_iterations": args.num_iterations
    }
    
    with open(os.path.join(debug_dir, f"summary_stats_{task}{suffix}.json"), 
              "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

logging.info("所有任务完成!")
