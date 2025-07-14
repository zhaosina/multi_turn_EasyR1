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
import numpy as np

def get_system_prompt():
    """返回OS-Atlas模型的完整系统提示词"""
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
    """解析模型响应获取相对坐标"""
    resp = resp.replace('<|im_end|>', '').strip()
    
    # 匹配标准 CLICK 格式: CLICK <point>[[x, y]]</point>
    m = re.search(r'CLICK\s+<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return [x / 1000.0, y / 1000.0]
    
    # 匹配带actions前缀的格式
    m = re.search(r'actions?:\s*CLICK\s+<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return [x / 1000.0, y / 1000.0]
    
    # 匹配不带CLICK前缀的格式
    m = re.search(r'<point>\[\[(\d+),\s*(\d+)\]\]</point>', resp, flags=re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return [x / 1000.0, y / 1000.0]
    
    raise ValueError("no_coord")

def process_single_iteration_with_system_prompt(model, processor, image, task_instruction, iteration):
    """使用系统提示词处理单次迭代"""
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
    
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    
    return response

def draw_point_on_image(image, x_rel, y_rel, iteration, color=None):
    """在图像上绘制预测点"""
    img_w, img_h = image.size
    x_pixel = int(x_rel * img_w)
    y_pixel = int(y_rel * img_h)
    
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
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

def convert_bbox_to_relative(bbox_xyxy, img_w, img_h):
    """将绝对坐标转换为相对坐标 [0-1] - ScreenSpot-Pro已经是[x1,y1,x2,y2]格式"""
    x1, y1, x2, y2 = bbox_xyxy
    return [x1/img_w, y1/img_h, x2/img_w, y2/img_h]

def validate_screenspot_pro_format(json_path):
    """验证 ScreenSpot-Pro 数据格式"""
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
        
        # 验证 bbox 格式 - 应该是 [x1, y1, x2, y2]
        bbox = sample["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            return False, f"bbox 格式错误: {bbox}"
        
        # 验证 img_size 格式
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
    """发现可用的任务文件"""
    tasks = {}
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        # 解析文件名获取任务信息
        base_name = json_file.replace('.json', '')
        
        # 尝试解析 application_platform.json 格式
        if '_' in base_name:
            parts = base_name.split('_')
            if len(parts) >= 2:
                app = '_'.join(parts[:-1])  # 应用名（可能包含下划线）
                platform = parts[-1]       # 平台名
                
                task_key = f"{app}_{platform}"
                tasks[task_key] = {
                    'file': json_file,
                    'application': app,
                    'platform': platform,
                    'display_name': f"{app} ({platform})"
                }
        else:
            # 单一名称文件
            tasks[base_name] = {
                'file': json_file,
                'application': base_name,
                'platform': 'unknown',
                'display_name': base_name
            }
    
    return tasks

def main():
    parser = argparse.ArgumentParser(description="ScreenSpot-Pro Model Testing")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True,
                       help='Task to test (app_platform format, e.g., davinci_macos, or "all")')
    parser.add_argument('--log_file', type=str, default='eval_screenspot_pro.log')
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
    
    # 加载模型
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # 发现可用任务
    available_tasks = discover_available_tasks(args.screenspot_test)
    logging.info(f"发现 {len(available_tasks)} 个可用任务:")
    for task_key, task_info in available_tasks.items():
        logging.info(f"  {task_key}: {task_info['display_name']}")
    
    # 确定要测试的任务
    if args.task == "all":
        tasks_to_test = list(available_tasks.keys())
    else:
        if args.task in available_tasks:
            tasks_to_test = [args.task]
        else:
            logging.error(f"任务 '{args.task}' 不存在，可用任务: {list(available_tasks.keys())}")
            return
    
    # 创建输出目录
    debug_dir = os.path.join(args.screenspot_test, "debug_logs")
    iteration_imgs_dir = os.path.join(debug_dir, "iteration_images")
    os.makedirs(debug_dir, exist_ok=True)
    if args.save_iteration_images:
        os.makedirs(iteration_imgs_dir, exist_ok=True)
    
    # 处理每个任务[3][4][5][6]
    for task_key in tasks_to_test:
        task_info = available_tasks[task_key]
        logging.info(f"\n=== 开始处理任务: {task_info['display_name']} ===")
        
        json_path = os.path.join(args.screenspot_test, task_info['file'])
        
        # 验证数据格式
        is_valid, msg = validate_screenspot_pro_format(json_path)
        if not is_valid:
            logging.error(f"[{task_key}] 数据格式验证失败: {msg}")
            continue
        
        # 加载数据
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"任务 `{task_key}` 共 {len(data)} 条，使用 {args.num_iterations} 次迭代")
        
        # 统计变量
        all_outputs = []
        wrong_format = []
        no_hit = []
        num_action = 0
        corr_action = 0
        
        # 原始模型准确率统计
        original_model_correct = 0
        original_model_total = 0
        
        iteration_stats = {i: {"correct": 0, "total": 0} for i in range(args.num_iterations)}
        
        for idx, item in tqdm(enumerate(data), total=len(data), desc=f"处理 {task_key}"):
            # 验证必需字段
            required_fields = ["img_filename", "bbox", "instruction", "img_size"]
            if not all(field in item for field in required_fields):
                logging.warning(f"[{task_key}][{idx}] 缺少必需字段")
                wrong_format.append({"idx": idx, "reason": "missing_fields"})
                continue
            
            # 构建图像路径 - ScreenSpot-Pro 图像在子文件夹中
            img_file = os.path.join(args.screenspot_imgs, item["img_filename"])
            if not os.path.exists(img_file):
                logging.warning(f"[{task_key}][{idx}] 缺失图片 {img_file}")
                wrong_format.append({"idx": idx, "reason": "img_missing"})
                continue
            
            try:
                # ScreenSpot-Pro 的 bbox 已经是 [x1, y1, x2, y2] 格式，无需转换
                bbox_xyxy = item["bbox"]
                img_size = item["img_size"]  # [width, height]
                img_w, img_h = img_size
                
                # 加载图像并验证尺寸
                original_image = Image.open(img_file).convert("RGB")
                actual_w, actual_h = original_image.size
                
                # 检查尺寸是否匹配
                if actual_w != img_w or actual_h != img_h:
                    logging.warning(f"[{task_key}][{idx}] 图像尺寸不匹配: "
                                  f"标注{img_w}x{img_h} vs 实际{actual_w}x{actual_h}")
                    img_w, img_h = actual_w, actual_h  # 使用实际尺寸
                
                # 转换为相对坐标
                bbox_gt_rel = convert_bbox_to_relative(bbox_xyxy, img_w, img_h)
                
                instr = item["instruction"]
                
                # 记录详细信息（仅前5个样本）
                if idx < 5:
                    logging.info(f"[{task_key}][{idx}] 应用: {item.get('application')}, "
                               f"平台: {item.get('platform')}")
                    logging.info(f"[{task_key}][{idx}] 图像尺寸: {img_w}x{img_h}")
                    logging.info(f"[{task_key}][{idx}] bbox(xyxy): {bbox_xyxy}")
                    logging.info(f"[{task_key}][{idx}] 相对坐标: {bbox_gt_rel}")
                    logging.info(f"[{task_key}][{idx}] 指令: {instr}")
                
                iteration_results = []
                current_image = original_image.copy()
                final_hit = False
                first_iteration_hit = False
                
                # 多次迭代预测
                for iteration in range(args.num_iterations):
                    if final_hit:
                        logging.info(f"[{task_key}][{idx}] 已在迭代 {iteration} 前命中，跳过后续迭代")
                        break
                    
                    try:
                        response = process_single_iteration_with_system_prompt(
                            model, processor, current_image, instr, iteration
                        )
                        
                        print(f"Iteration {iteration + 1} Response: {response}")
                        pred_rel = parse_response_with_system_prompt(response, img_w, img_h)
                        
                        # 检查命中
                        hit = (bbox_gt_rel[0] <= pred_rel[0] <= bbox_gt_rel[2] and 
                               bbox_gt_rel[1] <= pred_rel[1] <= bbox_gt_rel[3])
                        
                        # 记录第一次迭代结果
                        if iteration == 0:
                            original_model_total += 1
                            if hit:
                                first_iteration_hit = True
                                original_model_correct += 1
                                logging.info(f"[{task_key}][{idx}] 第一次迭代命中! (原始模型成功)")
                            else:
                                logging.info(f"[{task_key}][{idx}] 第一次迭代未命中 (原始模型失败)")
                        
                        iteration_results.append({
                            "iteration": iteration,
                            "response": response,
                            "prediction": pred_rel,
                            "hit": hit,
                            "is_first_iteration": iteration == 0,
                            "first_iteration_success": first_iteration_hit if iteration == 0 else None
                        })
                        
                        # 在图像上标记预测点
                        current_image = draw_point_on_image(current_image, pred_rel[0], pred_rel[1], iteration)
                        
                        # 保存迭代图像
                        if args.save_iteration_images:
                            iter_img_path = os.path.join(iteration_imgs_dir, 
                                                       f"{task_key}_{idx}_iter_{iteration+1}.png")
                            current_image.save(iter_img_path)
                        
                        # 更新统计
                        iteration_stats[iteration]["total"] += 1
                        if hit:
                            iteration_stats[iteration]["correct"] += 1
                            final_hit = True
                            logging.info(f"[{task_key}][{idx}] 迭代 {iteration+1}: 命中! 停止后续迭代")
                        else:
                            logging.info(f"[{task_key}][{idx}] 迭代 {iteration+1}: 未命中，坐标: {pred_rel}")
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
                            logging.info(f"[{task_key}][{idx}] 第一次迭代错误 (原始模型失败)")
                
                # 评估最终结果
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
                                "first_iteration_success": first_iteration_hit,
                                "application": item.get('application'),
                                "platform": item.get('platform'),
                                "ui_type": item.get('ui_type')
                            })
                        
                        # 更新日志
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
                        "ui_type": item.get('ui_type'),
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
        
        # 保存结果文件
        with open(os.path.join(debug_dir, f"all_outputs_{task_key}_multishot.json"), 
                  "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(debug_dir, f"wrong_format_{task_key}_multishot.json"), 
                  "w", encoding="utf-8") as f:
            json.dump(wrong_format, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(debug_dir, f"no_hit_{task_key}_multishot.json"), 
                  "w", encoding="utf-8") as f:
            json.dump(no_hit, f, ensure_ascii=False, indent=2)
        
        # 计算最终统计结果
        original_model_acc = original_model_correct / max(original_model_total, 1)
        multi_iteration_acc = corr_action / max(num_action, 1)
        
        # 输出详细统计
        logging.info(f"\n{'='*60}")
        logging.info(f"{task_info['display_name']} 最终统计结果:")
        logging.info(f"{'='*60}")
        logging.info(f"原始模型准确率 (仅第1次迭代成功): {original_model_acc:.3f} ({original_model_correct}/{original_model_total})")
        logging.info(f"多迭代总准确率 (任意迭代成功): {multi_iteration_acc:.3f} ({corr_action}/{num_action})")
        logging.info(f"准确率提升: {multi_iteration_acc - original_model_acc:.3f}")
        logging.info(f"格式错误: {len(wrong_format)}, 最终未命中: {len(no_hit)}")
        
        # 每个迭代的详细统计
        logging.info(f"\n各迭代详细统计:")
        for i in range(args.num_iterations):
            stats = iteration_stats[i]
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                logging.info(f"迭代 {i+1}: {acc:.3f} ({stats['correct']}/{stats['total']})")
        
        # 保存统计摘要
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

if __name__ == "__main__":
    main()
