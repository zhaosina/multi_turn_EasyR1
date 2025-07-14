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

def draw_point_on_image(image, x_pred, y_pred, iteration, img_w, img_h, color=None):
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    point_color = color or colors[iteration % len(colors)]
    radius = 8
    x_pixel = int(x_pred * img_w) if x_pred <= 1.0 else int(x_pred)
    y_pixel = int(y_pred * img_h) if y_pred <= 1.0 else int(y_pred)
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

def validate_screenspot_pro_format(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data:
            return False, "空数据文件"
        sample = data[0]
        required_fields = ["img_filename", "bbox", "instruction", "application", "platform", "img_size"]
        for field in required_fields:
            if field not in sample:
                return False, f"缺少字段: {field}"
        bbox = sample["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return False, f"bbox 格式错误: {bbox}"
        img_size = sample["img_size"]
        if not (isinstance(img_size, list) and len(img_size) == 2):
            return False, f"img_size 格式错误: {img_size}"
        logging.info(f"数据格式验证通过，样本数: {len(data)}")
        logging.info(f"应用: {sample['application']}, 平台: {sample['platform']}")
        logging.info(f"bbox 格式: {bbox} (x1,y1,x2,y2)")
        logging.info(f"图像尺寸: {img_size}")
        return True, "格式正确"
    except Exception as e:
        return False, f"文件读取错误: {str(e)}"

def discover_available_tasks(data_dir):
    tasks = {}
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    for json_file in json_files:
        base_name = json_file.replace('.json', '')
        if '_' in base_name:
            parts = base_name.split('_')
            if len(parts) >= 2:
                app = '_'.join(parts[:-1])
                platform = parts[-1]
                task_key = f"{app}_{platform}"
                tasks[task_key] = {
                    'file': json_file,
                    'application': app,
                    'platform': platform,
                    'display_name': f"{app} ({platform})"
                }
        else:
            tasks[base_name] = {
                'file': json_file,
                'application': base_name,
                'platform': 'unknown',
                'display_name': base_name
            }
    return tasks

parser = argparse.ArgumentParser()
parser.add_argument('--qwen_path', type=str, required=True)
parser.add_argument('--screenspot_imgs', type=str, required=True)
parser.add_argument('--screenspot_test', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--log_file', type=str, default='eval.log')
parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--save_iteration_images', action='store_true')
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

available_tasks = discover_available_tasks(args.screenspot_test)
logging.info(f"发现 {len(available_tasks)} 个可用任务:")
for task_key, task_info in available_tasks.items():
    logging.info(f"  {task_key}: {task_info['display_name']}")

if args.task == "all":
    tasks_to_test = list(available_tasks.keys())
else:
    if args.task in available_tasks:
        tasks_to_test = [args.task]
    else:
        logging.error(f"任务 '{args.task}' 不存在，可用任务: {list(available_tasks.keys())}")
        exit()

debug_dir = os.path.join(args.screenspot_test, "debug_logs")
iteration_imgs_dir = os.path.join(debug_dir, "iteration_images")
os.makedirs(debug_dir, exist_ok=True)
if args.save_iteration_images:
    os.makedirs(iteration_imgs_dir, exist_ok=True)

for task_key in tasks_to_test:
    task_info = available_tasks[task_key]
    logging.info(f"\n=== 开始处理任务: {task_info['display_name']} ===")
    
    json_path = os.path.join(args.screenspot_test, task_info['file'])
    
    is_valid, msg = validate_screenspot_pro_format(json_path)
    if not is_valid:
        logging.error(f"[{task_key}] 数据格式验证失败: {msg}")
        continue
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logging.info(f"任务 `{task_key}` 共 {len(data)} 条，使用 {args.num_iterations} 次迭代")
    prompt_tpl = ('In this UI screenshot, what is the position of the element '
                  'corresponding to the command "{}" (with point)?')

    all_outputs = []
    wrong_format = []
    no_hit = []
    num_action = 0
    corr_action = 0
    original_model_correct = 0
    original_model_total = 0
    iteration_stats = {i: {"correct": 0, "total": 0} for i in range(args.num_iterations)}

    for idx, item in tqdm(enumerate(data), total=len(data), desc=f"处理 {task_key}"):
        required_fields = ["img_filename", "bbox", "instruction", "img_size"]
        if not all(field in item for field in required_fields):
            logging.warning(f"[{task_key}][{idx}] 缺少必需字段")
            wrong_format.append({"idx": idx, "reason": "missing_fields"})
            continue

        img_file = os.path.join(args.screenspot_imgs, item["img_filename"])
        if not os.path.exists(img_file):
            logging.warning(f"[{task_key}][{idx}] 缺失图片 {img_file}")
            wrong_format.append({"idx": idx, "reason": "img_missing"})
            continue

        try:
            bbox_xyxy = item["bbox"]
            img_size = item["img_size"]
            img_w, img_h = img_size
            
            original_image = Image.open(img_file).convert("RGB")
            actual_w, actual_h = original_image.size
            
            if actual_w != img_w or actual_h != img_h:
                logging.warning(f"[{task_key}][{idx}] 图像尺寸不匹配: "
                              f"标注{img_w}x{img_h} vs 实际{actual_w}x{actual_h}")
                img_w, img_h = actual_w, actual_h
                scale_x = actual_w / img_size[0]
                scale_y = actual_h / img_size[1]
                bbox_xyxy = [int(bbox_xyxy[0] * scale_x), int(bbox_xyxy[1] * scale_y),
                            int(bbox_xyxy[2] * scale_x), int(bbox_xyxy[3] * scale_y)]
            
            instr = item["instruction"]
            bbox_gt = bbox_xyxy
            
            if idx < 5:
                logging.info(f"[{task_key}][{idx}] 应用: {item.get('application')}, "
                           f"平台: {item.get('platform')}")
                logging.info(f"[{task_key}][{idx}] 图像尺寸: {img_w}x{img_h}")
                logging.info(f"[{task_key}][{idx}] bbox(xyxy): {bbox_gt}")
                logging.info(f"[{task_key}][{idx}] 指令: {instr}")

            iteration_results = []
            current_image = original_image.copy()
            final_hit = False
            first_iteration_hit = False

            for iteration in range(args.num_iterations):
                if final_hit:
                    logging.info(f"[{task_key}][{idx}] 已在迭代 {iteration} 前命中，跳过后续迭代")
                    break

                try:
                    if iteration == 0:
                        prompt = ('Only return "<point>[x, y]</point>".\n\n' + prompt_tpl.format(instr))
                    else:
                        prompt_iter = 'In this UI screenshot with previous attempts marked, what is the position of the element corresponding to the command "{}" (with point)?'
                        prompt = ('Only return "<point>[x, y]</point>". Previous attempts are shown as colored dots. Learn from them.\n\n' + prompt_iter.format(instr))
                    
                    messages = [{"role": "user",
                                 "content": [{"type": "image", "image": current_image},
                                            {"type": "text",  "text": prompt}]}]

                    text_in = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text_in], images=[current_image], return_tensors="pt", padding=True)
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(model.device)

                    out_ids = model.generate(**inputs)
                    response = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
                    
                    logging.info(f"[{task_key}][{idx}] 迭代 {iteration + 1} 响应: {response[:100]}...")

                    def parse(resp: str):
                        m = re.search(r'<points[^>]*x1\s*=\s*[\'"]?([\d.]+)[\'"]?[^>]*'
                                      r'y1\s*=\s*[\'"]?([\d.]+)[\'"]?', resp, flags=re.I)
                        if m:
                            return [float(m.group(1)) / img_w, float(m.group(2)) / img_h]
                        if 'box' in resp.lower():
                            b = extract_bbox(resp)
                            return [((b[0][0] + b[1][0]) / 2) / img_w,
                                    ((b[0][1] + b[1][1]) / 2) / img_h]
                        m = re.search(r'[\[\(]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\]\)]', resp)
                        if m:
                            return [float(m.group(1)), float(m.group(2))]
                        raise ValueError("no_coord")

                    pred = parse(response)
                    
                    if pred[0] <= 1.0 and pred[1] <= 1.0:
                        pred_abs = [pred[0] * img_w, pred[1] * img_h]
                    else:
                        pred_abs = pred
                    
                    hit = bbox_gt[0] <= pred_abs[0] <= bbox_gt[2] and \
                          bbox_gt[1] <= pred_abs[1] <= bbox_gt[3]

                    if iteration == 0:
                        original_model_total += 1
                        if hit:
                            first_iteration_hit = True
                            original_model_correct += 1
                            logging.info(f"[{task_key}][{idx}] 第一次迭代命中! 坐标: {pred}")
                        else:
                            logging.info(f"[{task_key}][{idx}] 第一次迭代未命中, 坐标: {pred}")

                    iteration_results.append({
                        "iteration": iteration,
                        "response": response,
                        "prediction": pred,
                        "hit": hit,
                        "is_first_iteration": iteration == 0,
                        "first_iteration_success": first_iteration_hit if iteration == 0 else None
                    })

                    current_image = draw_point_on_image(current_image, pred[0], pred[1], iteration, img_w, img_h)

                    if args.save_iteration_images:
                        iter_img_path = os.path.join(iteration_imgs_dir, 
                                                   f"{task_key}_{idx}_iter_{iteration+1}.png")
                        current_image.save(iter_img_path)

                    iteration_stats[iteration]["total"] += 1
                    if hit:
                        iteration_stats[iteration]["correct"] += 1
                        final_hit = True
                        logging.info(f"[{task_key}][{idx}] 迭代 {iteration+1}: 命中! 停止后续迭代")
                    else:
                        logging.info(f"[{task_key}][{idx}] 迭代 {iteration+1}: 未命中，坐标: {pred}")
                        if iteration == args.num_iterations - 1:
                            logging.info(f"[{task_key}][{idx}] 所有 {args.num_iterations} 次迭代均未命中")

                except Exception as e:
                    logging.error(f"[{task_key}][{idx}] 迭代 {iteration+1} 错误: {e}")
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
                        logging.info(f"[{task_key}][{idx}] 第一次迭代错误")

            try:
                num_action += 1
                valid_predictions = [r["prediction"] for r in iteration_results if "prediction" in r]

                if valid_predictions:
                    final_pred = valid_predictions[-1]
                    any_hit = any(r.get("hit", False) for r in iteration_results)

                    if any_hit:
                        corr_action += 1
                    else:
                        no_hit.append({
                            "idx": idx,
                            "final_pred": final_pred,
                            "all_predictions": valid_predictions,
                            "gt": [(bbox_gt[0]+bbox_gt[2])/2, (bbox_gt[1]+bbox_gt[3])/2],
                            "bbox_gt": bbox_gt,
                            "img_size": [img_w, img_h],
                            "iterations_attempted": len(iteration_results),
                            "first_iteration_success": first_iteration_hit,
                            "application": item.get('application'),
                            "platform": item.get('platform')
                        })

                    original_acc = original_model_correct / max(original_model_total, 1)
                    multi_iter_acc = corr_action / num_action

                    logging.info(f"[{task_key}][{idx}] 最终结果: {'命中' if any_hit else '未命中'} "
                               f"使用了 {len(iteration_results)} 次迭代")
                    logging.info(f"[{task_key}][{idx}] 当前统计 - "
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
                    "application": item.get('application'),
                    "platform": item.get('platform'),
                    "bbox_gt": bbox_gt,
                    "img_size": [img_w, img_h],
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
                logging.error(f"[{task_key}][{idx}] 整体错误 ({e})")

        except Exception as e:
            logging.error(f"[{task_key}][{idx}] 样本处理错误: {e}")
            wrong_format.append({"idx": idx, "reason": f"processing_error: {str(e)}"})
            continue

    with open(os.path.join(debug_dir, f"all_outputs_{task_key}_multishot.json"), 
              "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    with open(os.path.join(debug_dir, f"wrong_format_{task_key}_multishot.json"), 
              "w", encoding="utf-8") as f:
        json.dump(wrong_format, f, ensure_ascii=False, indent=2)
    with open(os.path.join(debug_dir, f"no_hit_{task_key}_multishot.json"), 
              "w", encoding="utf-8") as f:
        json.dump(no_hit, f, ensure_ascii=False, indent=2)

    original_model_acc = original_model_correct / max(original_model_total, 1)
    multi_iteration_acc = corr_action / max(num_action, 1)

    logging.info(f"\n{'='*60}")
    logging.info(f"{task_info['display_name']} 最终统计结果:")
    logging.info(f"{'='*60}")
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
        "task": task_key,
        "task_info": task_info,
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

    with open(os.path.join(debug_dir, f"summary_stats_{task_key}.json"), 
              "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

logging.info("所有任务完成!")
