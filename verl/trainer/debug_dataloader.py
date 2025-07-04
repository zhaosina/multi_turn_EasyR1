import torch
import os
import json
import logging
from PIL import Image
from typing import List, Dict, Any
from transformers import AutoProcessor, AutoTokenizer
from functools import partial

# 确保这些导入路径是正确的，你可能需要根据项目结构进行调整
# 假设此脚本与 verl 目录同级
from verl.trainer.data_loader import MGRPODataset, mgrpo_v_collate_fn
from verl.trainer.config import PPOConfig 

def debug_the_dataloader():
    """
    一个用于深入调试数据加载和整理过程的独立脚本。
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- 1. 配置 (请根据你的系统调整路径) ---
    model_path = "/home/zhq/workdir/GUI/Qwen2.5-VL-3B-Instruct"
    processed_data_path = "/home/zhq/workdir/GUI/screenspot-v2/processed/val.jsonl" 
    image_dir = "/home/zhq/workdir/GUI/screenspot-v2/screenspotv2_image"
    batch_size = 4 # 使用一个小的批次大小进行调试

    # --- 2. 加载 Processor 和 Tokenizer ---
    logging.info(f"正在从 {model_path} 加载 processor 和 tokenizer...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logging.info("Processor 和 tokenizer 加载成功。")
    except Exception as e:
        logging.error(f"加载 processor/tokenizer 失败: {e}")
        return

    # --- 3. 创建数据集实例 ---
    logging.info("正在初始化 MGRPODataset...")
    try:
        # 确保传入所有必要的参数
        dataset = MGRPODataset(
            data_path=processed_data_path,
            processor=processor,
            tokenizer=tokenizer,
            image_dir=image_dir,
        )
        logging.info(f"数据集创建成功，包含 {len(dataset)} 个样本。")
    except Exception as e:
        logging.error(f"创建数据集失败: {e}")
        return

    # --- 4. 手动模拟一个批次的创建过程 ---
    logging.info(f"\n{'='*20} 开始手动模拟批次创建 {'='*20}")
    if len(dataset) < batch_size:
        logging.error(f"数据集太小，无法创建批次。需要至少 {batch_size} 个样本。")
        return

    raw_batch = [dataset[i] for i in range(batch_size)]
    raw_batch = [item for item in raw_batch if item]

    if len(raw_batch) != batch_size:
        logging.error("无法创建完整的批次，部分样本加载失败。")
        return

    logging.info(f"--- 步骤 4.1: 检查从 __getitem__ 返回的单个样本数据 ---")
    # ... (这部分日志可以保持不变)

    # --- 5. 调用我们正在调试的 collate_fn ---
    logging.info(f"--- 步骤 4.2: 调用 mgrpo_v_collate_fn 并检查最终输出 ---")
    try:
        # --- CORRECTED: 使用 functools.partial 绑定所有需要的参数 ---
        collate_wrapper = partial(mgrpo_v_collate_fn, processor=processor, tokenizer=tokenizer)
        final_batch_dict = collate_wrapper(raw_batch)
        
        logging.info("mgrpo_v_collate_fn 调用成功。最终批次内容如下:")
        for key, value in final_batch_dict.items():
             if isinstance(value, torch.Tensor):
                print(f"键: '{key}', 形状: {value.shape}, 类型: {value.dtype}")
             else:
                print(f"键: '{key}', 类型: {type(value)}")
        
        logging.info("\n--- 最终诊断 ---")
        input_ids_shape = final_batch_dict.get('input_ids').shape
        pixel_values_shape = final_batch_dict.get('pixel_values').shape

        print(f"Input IDs 的最终形状: {input_ids_shape}")
        print(f"Pixel Values 的最终形状: {pixel_values_shape}")

        if len(pixel_values_shape) == 4 and pixel_values_shape[0] == batch_size:
             logging.info("✅ 成功: 'pixel_values' 的形状和批次大小看起来完全正确！")
        else:
             logging.error(f"❌ 失败: 'pixel_values' 的最终形状不正确。预期批次大小为 {batch_size}，但张量形状为 {pixel_values_shape}。")

    except Exception as e:
        logging.error(f"在调用 collate_fn 时发生错误: {e}", exc_info=True)

if __name__ == "__main__":
    debug_the_dataloader()
