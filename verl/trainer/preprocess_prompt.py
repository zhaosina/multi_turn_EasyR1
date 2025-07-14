import json
import os
import argparse
import random
from tqdm import tqdm
import logging

# --- 从您的测试脚本中直接引入提示词生成函数 ---

def get_system_prompt():
    """获取系统提示"""
    return """You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions' coordinates in specific format that best fit their needs:
CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>

Your current task instruction, action grounding history, and associated screenshot are as follows:
Screenshot: """

def get_task_prompts():
    """获取任务提示模板"""
    # 对于在线RL，我们主要关注初始提示，后续的纠正行为是模型需要学习的
    return ["Task instruction: {}\nHistory: null"]

# --- 主处理函数 ---

def create_grpo_dataset_with_system_prompt(
    data_dir: str,
    output_dir: str,
    train_split_ratio: float = 0.98,
    seed: int = 42
):
    """
    将 screenspot-v2 数据集转换为 GRPO 训练格式。
    每个样本的 'prompt' 字段都将包含完整的系统提示和任务指令。
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    source_files = [
        "screenspot_desktop_v2.json",
        "screenspot_mobile_v2.json",
        "screenspot_web_v2.json"
    ]

    all_samples = []
    logging.info("Loading and combining source JSON files...")
    for file_name in source_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_samples.extend(data)
        else:
            logging.warning(f"File not found, skipping: {file_path}")

    logging.info(f"Total samples combined: {len(all_samples)}")
    
    # 获取提示词模板
    system_prompt = get_system_prompt()
    task_prompt_template = get_task_prompts()[0]

    processed_data = []
    for i, item in enumerate(tqdm(all_samples, desc="Processing samples for GRPO")):
        if "instruction" not in item or "bbox" not in item or "img_filename" not in item:
            continue

        instruction = item["instruction"]

        # 1. 构建完整的、包含系统指令的提示
        task_prompt = task_prompt_template.format(instruction)
        full_prompt = f"{system_prompt}\n\n{task_prompt}"

        processed_data.append({
            "trajectory_id": i,
            "turn_id": 0,
            "prompt": full_prompt, # <-- 核心变化在这里
            "image": item["img_filename"],
            "ground_truth_bbox": item["bbox"],
        })

    # 3. 分割并保存数据集
    random.shuffle(processed_data)
    split_index = int(len(processed_data) * train_split_ratio)
    train_data = processed_data[:split_index]
    val_data = processed_data[split_index:]

    def save_as_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"Saved {len(data)} samples to {path}")

    save_as_jsonl(train_data, os.path.join(output_dir, "train_with_prompt.jsonl"))
    save_as_jsonl(val_data, os.path.join(output_dir, "val_with_prompt.jsonl"))
    
    logging.info("Preprocessing for GRPO with system prompts complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess screenspot-v2 data for GRPO training with system prompts.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to the root directory of the screenspot-v2 dataset."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the processed .jsonl files."
    )
    args = parser.parse_args()
    
    create_grpo_dataset_with_system_prompt(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
