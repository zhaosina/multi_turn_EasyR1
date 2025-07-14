import torch
import argparse
import os
import json
import logging
import re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration
from process_utils import extract_bbox
import numpy as np

def parse_response(resp: str, img_w: int, img_h: int):
    m = re.search(
        r'(?:<)?point[^>]*>\[\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\]</point>',
        resp, flags=re.I
    )
    if m:
        return [float(m.group(1))/1000.0, float(m.group(2))/1000.0]

    m = re.search(
        r'<point[^>]*\s+x1\s*=\s*[\'"]?\(?\s*([\d.]+)\s*,\s*([\d.]+)\)?',
        resp, flags=re.I
    )
    if m:
        return [float(m.group(1))/1000.0, float(m.group(2))/1000.0]

    m = re.search(
        r'<point[^>]*\s+x1\s*=\s*[\'"]?([\d.]+)[\'"]?[^>]*'
        r'y1\s*=\s*[\'"]?([\d.]+)',
        resp, flags=re.I
    )
    if m:
        return [float(m.group(1))/1000.0, float(m.group(2))/1000.0]

    m = re.search(
        r'<x1\s*=\s*[\'"]?([\d.]+)[\'"]?\s*'
        r'y1\s*=\s*[\'"]?([\d.]+)',
        resp, flags=re.I
    )
    if m:
        x = float(m.group(1))
        y = float(m.group(2))
        return [x / 1000.0, y / 1000.0]

    if 'box' in resp.lower():
        b = extract_bbox(resp)
        return [((b[0][0] + b[1][0]) / 2) / 1000.0,
                ((b[0][1] + b[1][1]) / 2) / 1000.0]

    m = re.search(r'[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\]\)]', resp)
    if m:
        x, y = map(float, [m.group(1), m.group(2)])
        if x > 1000 or y > 1000:
            return [x / img_w, y / img_h]
        return [x / 1000.0, y / 1000.0]

    raise ValueError("no_coord")

def draw_point_on_image(image, x_rel, y_rel, iteration, color=None):
    img_w, img_h = image.size
    x_pixel = int(x_rel * img_w)
    y_pixel = int(y_rel * img_h)
    
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红、绿、蓝
    point_color = color or colors[iteration % len(colors)]
    
    radius = 8
    draw.ellipse([x_pixel-radius, y_pixel-radius, x_pixel+radius, y_pixel+radius], 
                 fill=point_color, outline=(255, 255, 255), width=2)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((x_pixel+radius+2, y_pixel-radius), f"#{iteration+1}", 
              fill=point_color, font=font)
    
    return new_image

def evaluate_predictions_with_gt(predictions, bbox_gt_rel):
    best_pred = None
    best_score = float('inf')
    
    gt_center_x = (bbox_gt_rel[0] + bbox_gt_rel[2]) / 2
    gt_center_y = (bbox_gt_rel[1] + bbox_gt_rel[3]) / 2
    
    for pred in predictions:
        distance = np.sqrt((pred[0] - gt_center_x)**2 + (pred[1] - gt_center_y)**2)
        if distance < best_score:
            best_score = distance
            best_pred = pred
    
    return best_pred, best_score

def get_system_prompt():
    return """You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
       
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]
    
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.
Custom Action 1: LONG_PRESS 
    - purpose: Long press at the specified position.
    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_PRESS <point>[[101, 872]]</point>
       
Custom Action 2: OPEN_APP
    - purpose: Open the specified application.
    - format: OPEN_APP [app_name]
    - example usage: OPEN_APP [Google Chrome]

Custom Action 3: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 4: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 5: PRESS_RECENT
    - purpose: Press the recent button to view or switch between recently used applications.
    - format: PRESS_RECENT
    - example usage: PRESS_RECENT

Custom Action 6: ENTER
    - purpose: Press the enter button.
    - format: ENTER
    - example usage: ENTER

Custom Action 7: WAIT
    - purpose: Wait for the screen to load.
    - format: WAIT
    - example usage: WAIT

Custom Action 8: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
Thoughts: Clearly outline your reasoning process for current step.
Actions: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

Your current task instruction, action history, and associated screenshot are as follows:
Screenshot: """

def get_task_prompts():
    return [
        "Task instruction: {}\nHistory: null",
        "Task instruction: {}\nHistory: Previous attempts marked with colored dots on the screenshot.",
        "Task instruction: {}\nHistory: Multiple previous attempts shown as colored markers. Consider previous predictions when making your decision."
    ]

def parse_response_with_system_prompt(resp: str, img_w: int, img_h: int):
   
    resp = resp.replace('<|im_end|>', '').strip()
    
    m = re.search(r'CLICK\s+<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return [x / 1000.0, y / 1000.0]
    
    m = re.search(r'actions?:\s*CLICK\s+<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return [x / 1000.0, y / 1000.0]
    
    m = re.search(r'<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return [x / 1000.0, y / 1000.0]
    
    raise ValueError("no_coord")


def process_single_iteration_with_system_prompt(model, processor, image, task_instruction, iteration):

    temp_img_path = f"temp_iteration_{iteration}.png"
    image.save(temp_img_path)

    sys_prompt = get_system_prompt()
    task_prompts = get_task_prompts()
    task_prompt = task_prompts[min(iteration, len(task_prompts) - 1)].format(task_instruction)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": sys_prompt,
                },
                {
                    "type": "image",
                    "image": temp_img_path,
                },
                {
                    "type": "text", 
                    "text": task_prompt
                },
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    response = output_text[0] if output_text else ""
    
    # 清理临时文件
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    return response

def convert_bbox_format(bbox_xywh):
    x, y, width, height = bbox_xywh
    x1, y1 = x, y
    x2, y2 = x + width, y + height
    return [x1, y1, x2, y2]

def convert_bbox_to_relative(bbox_xyxy, img_w, img_h):
    x1, y1, x2, y2 = bbox_xyxy
    return [x1/img_w, y1/img_h, x2/img_w, y2/img_h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--log_file', type=str, default='eval.log')
    parser.add_argument('--num_iterations', type=int, default=3, 
                       help='Number of iterations for multi-shot prediction')
    parser.add_argument('--save_iteration_images', action='store_true',
                       help='Save images with marked points for each iteration')
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
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    tasks = (["android", "forum", "gitlab", "ios", "macos", "shop", "tool", "windows"]
             if args.task == "all" else [args.task])
    
    debug_dir = os.path.join(args.screenspot_test, "debug_logs")
    iteration_imgs_dir = os.path.join(debug_dir, "iteration_images")
    os.makedirs(debug_dir, exist_ok=True)
    if args.save_iteration_images:
        os.makedirs(iteration_imgs_dir, exist_ok=True)
    
    
    for task in tasks:
        json_path = os.path.join(args.screenspot_test, f"screenspot_{task}.json")
        if not os.path.exists(json_path):
            logging.error(f"标注文件不存在：{json_path}")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"子任务 `{task}` 共 {len(data)} 条，使用 {args.num_iterations} 次迭代")
        
        # 统计变量
        all_outputs = []
        wrong_format = []
        no_hit = []
        num_action = 0
        corr_action = 0
        
        original_model_correct = 0
        original_model_total = 0
        
        iteration_stats = {i: {"correct": 0, "total": 0} for i in range(args.num_iterations)}
        
        for idx, item in tqdm(list(enumerate(data)), desc=f"{task}"):
            img_file = os.path.join(args.screenspot_imgs, item["img_filename"])
            if not os.path.exists(img_file):
                logging.warning(f"[{task}][{idx}] 缺失图片 {img_file}")
                wrong_format.append({"idx": idx, "reason": "img_missing"})
                continue
            
            bbox_gt_rel = item["bbox"]
            original_image = Image.open(img_file).convert("RGB")
            img_w, img_h = original_image.size
            instr = item["instruction"]
            
            iteration_results = []
            current_image = original_image.copy()
            final_hit = False
            first_iteration_hit = False  
            
            for iteration in range(args.num_iterations):
                if final_hit:
                    logging.info(f"[{task}][{idx}] 已在迭代 {iteration} 前命中，跳过后续迭代")
                    break
                    
                try:
                    response = process_single_iteration_with_system_prompt(
                        model, processor, current_image, instr, iteration
                    )                  
                    print(f"Iteration {iteration + 1} Response: {response}")                    
                    pred_rel = parse_response_with_system_prompt(response, img_w, img_h)                   
                    hit = (bbox_gt_rel[0] <= pred_rel[0] <= bbox_gt_rel[2] and 
                           bbox_gt_rel[1] <= pred_rel[1] <= bbox_gt_rel[3])
                    
                    # 记录第一次迭代结果
                    if iteration == 0:
                        original_model_total += 1
                        if hit:
                            first_iteration_hit = True
                            original_model_correct += 1
                            logging.info(f"[{task}][{idx}] 第一次迭代命中! (原始模型成功)")
                        else:
                            logging.info(f"[{task}][{idx}] 第一次迭代未命中 (原始模型失败)")
                    
                    iteration_results.append({
                        "iteration": iteration,
                        "response": response,
                        "prediction": pred_rel,
                        "hit": hit,
                        "is_first_iteration": iteration == 0,
                        "first_iteration_success": first_iteration_hit if iteration == 0 else None
                    })
                    
                    current_image = draw_point_on_image(current_image, pred_rel[0], pred_rel[1], iteration)
                    
                    if args.save_iteration_images:
                        iter_img_path = os.path.join(iteration_imgs_dir, f"{task}_{idx}_iter_{iteration+1}.png")
                        current_image.save(iter_img_path)
                    
                    iteration_stats[iteration]["total"] += 1
                    if hit:
                        iteration_stats[iteration]["correct"] += 1
                        final_hit = True
                        logging.info(f"[{task}][{idx}] 迭代 {iteration+1}: 命中! 停止后续迭代")
                    else:
                        logging.info(f"[{task}][{idx}] 迭代 {iteration+1}: 未命中，坐标: {pred_rel}")
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
                    
                    # 如果第一次迭代出错，统计到原始模型
                    if iteration == 0:
                        original_model_total += 1
                        logging.info(f"[{task}][{idx}] 第一次迭代错误 (原始模型失败)")

            try:
                num_action += 1
                valid_predictions = [r["prediction"] for r in iteration_results if "prediction" in r]
                
                if valid_predictions:
                    final_pred = valid_predictions[-1]
                    any_hit = any(r.get("hit", False) for r in iteration_results)
                    
                    if any_hit:
                        corr_action += 1
                    else:
                        gt_center_x = (bbox_gt_rel[0] + bbox_gt_rel[2]) / 2
                        gt_center_y = (bbox_gt_rel[1] + bbox_gt_rel[3]) / 2
                        no_hit.append({
                            "idx": idx,
                            "final_pred": final_pred,
                            "all_predictions": valid_predictions,
                            "gt": [gt_center_x, gt_center_y],
                            "bbox_gt": bbox_gt_rel,
                            "iterations_attempted": len(iteration_results),
                            "first_iteration_success": first_iteration_hit
                        })
                    
                    original_acc = original_model_correct / max(original_model_total, 1)
                    multi_iter_acc = corr_action / num_action
                    
                    logging.info(f"[{task}][{idx}] 最终结果: {'命中' if any_hit else '未命中'} "
                               f"使用了 {len(iteration_results)} 次迭代")
                    logging.info(f"[{task}][{idx}] 当前统计 - "
                               f"原始模型准确率: {original_acc:.3f} ({original_model_correct}/{original_model_total}), "
                               f"多迭代准确率: {multi_iter_acc:.3f} ({corr_action}/{num_action})")
                else:
                    wrong_format.append({
                        "idx": idx,
                        "reason": "no_valid_predictions",
                        "iteration_results": iteration_results
                    })
                
                all_outputs.append({
                    "idx": idx,
                    "instruction": instr,
                    "iteration_results": iteration_results,
                    "final_prediction": final_pred if valid_predictions else None,
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
        
        # 保存结果文件
        with open(os.path.join(debug_dir, f"all_outputs_{task}_multishot.json"), 
                  "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(debug_dir, f"wrong_format_{task}_multishot.json"), 
                  "w", encoding="utf-8") as f:
            json.dump(wrong_format, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(debug_dir, f"no_hit_{task}_multishot.json"), 
                  "w", encoding="utf-8") as f:
            json.dump(no_hit, f, ensure_ascii=False, indent=2)
        
        original_model_acc = original_model_correct / max(original_model_total, 1)
        multi_iteration_acc = corr_action / max(num_action, 1)
        
        logging.info(f"\n{'='*50}")
        logging.info(f"{task} 最终统计结果:")
        logging.info(f"{'='*50}")
        logging.info(f"原始模型准确率 (仅第1次迭代成功): {original_model_acc:.3f} ({original_model_correct}/{original_model_total})")
        logging.info(f"多迭代总准确率 (任意迭代成功): {multi_iteration_acc:.3f} ({corr_action}/{num_action})")
        logging.info(f"准确率提升: {multi_iteration_acc - original_model_acc:.3f}")
        logging.info(f"格式错误: {len(wrong_format)}, 最终未命中: {len(no_hit)}")
        logging.info(f"\n各迭代详细统计:")
        for i in range(args.num_iterations):
            stats = iteration_stats[i]
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                logging.info(f"迭代 {i+1}: {acc:.3f} ({stats['correct']}/{stats['total']})")
        
        summary_stats = {
            "task": task,
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
            "iteration_stats": iteration_stats
        }
        
        with open(os.path.join(debug_dir, f"summary_stats_{task}.json"), 
                  "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
