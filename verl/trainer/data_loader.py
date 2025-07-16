# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import os
import json
import logging
from PIL import Image
from typing import Optional, List, Dict, Any

from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig



import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

import json, logging, os
from functools import partial
from typing import Any, Dict, List, Tuple
import torchvision.transforms.v2 as T
import torch
import os
import json
import logging
import numpy as np # Import numpy
from PIL import Image
from typing import Optional, List, Dict, Any
from functools import partial

from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from .config import DataConfig

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
    # 原始图像尺寸 (W, H)
    original_w, original_h = img.size
    # 目标尺寸 (Tw, Th)
    target_w, target_h = target_size

    # 1. 计算缩放比 r
    ratio = min(target_w / original_w, target_h / original_h)
    
    # 计算缩放后的新尺寸 (rW, rH)
    new_w = int(original_w * ratio)
    new_h = int(original_h * ratio)
    
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 创建一个目标尺寸的灰色背景画布
    new_img = Image.new("RGB", target_size, fill_color)
    
    # 2. 计算填充偏移 (px, py)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    
    new_img.paste(img_resized, (paste_x, paste_y))
    
    # 3. 对 bounding box 进行坐标变换
    original_x, original_y, original_box_w, original_box_h = bbox_xywh
    
    # 应用您提供的公式
    # x' = x*r + px
    # y' = y*r + py
    # w' = w*r
    # h' = h*r
    transformed_x = original_x * ratio + paste_x
    transformed_y = original_y * ratio + paste_y
    transformed_w = original_box_w * ratio
    transformed_h = original_box_h * ratio
    
    transformed_bbox_xywh = [transformed_x, transformed_y, transformed_w, transformed_h]
    
    return new_img, transformed_bbox_xywh


class MGRPODataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        image_size: tuple = (448, 448),
        prompt_key: str = "prompt",
        image_key: str = "image",
        gt_bbox_key: str = "ground_truth_bbox",
        traj_id_key: str = "trajectory_id",
        turn_id_key: str = "turn_id",
        **kwargs, 
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.prompt_key = prompt_key
        self.image_key = image_key
        self.gt_bbox_key = gt_bbox_key
        self.traj_id_key = traj_id_key
        self.turn_id_key = turn_id_key
        
        logging.info(f"Loading MGRPO-V dataset from: {data_path}")
        self.data = self._load_data()
        logging.info(f"Loaded {len(self.data)} samples.")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Loads the JSONL dataset file."""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line in {self.data_path}")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item[self.image_key])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Could not load image {image_path}: {e}")
            return None # Return None to be filtered out in collate_fn

        # Get original bbox without transformation
        gt_bbox_xywh = item.get(self.gt_bbox_key)
        
        trajectory_id = item.get(self.traj_id_key)
        turn_id = item.get(self.turn_id_key)

        if gt_bbox_xywh is None or trajectory_id is None or turn_id is None:
            logging.error(f"Missing required MGRPO-V metadata in item {idx}.")
            return None

        return {
            "image": image,  # The original, variable-sized PIL Image
            "prompt": item[self.prompt_key],
            "gt_bboxes_xywh": gt_bbox_xywh, # The original bbox
            "trajectory_ids": trajectory_id,
            "turn_ids": turn_id,
        }

def mgrpo_v_collate_fn(
    batch: List[Dict[str, Any]],
    *,
    processor: ProcessorMixin,
    tokenizer: PreTrainedTokenizer,
    patch_size: int = 14 # Patch size for the Vision Transformer
) -> Dict[str, Any]:
    """
    [REWRITTEN] This version implements dynamic padding within each batch.
    It finds the max dimensions in the current batch, pads all images to that
    size, transforms bboxes, and then collates everything.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    batch_size = len(batch)

    # --- Step A: Find the maximum dimensions in the current batch ---
    max_w, max_h = 0, 0
    for sample in batch:
        w, h = sample["image"].size
        if w > max_w: max_w = w
        if h > max_h: max_h = h
    
    # --- Step B: Align dimensions to be divisible by the patch size ---
    target_w = (max_w + patch_size - 1) // patch_size * patch_size
    target_h = (max_h + patch_size - 1) // patch_size * patch_size
    target_size = (target_w, target_h)
    
    # --- Step C: Loop through samples, apply padding, and process ---
    proc_outs = []
    transformed_bboxes_list = []
    padded_images_for_opv = []

    for sample in batch:
        # Pad the image to the dynamic target size and transform its bbox
        padded_image, transformed_bbox = resize_and_transform_bbox(
            sample["image"], sample["gt_bboxes_xywh"], target_size
        )
        
        # Process the now-padded image with its corresponding text
        proc_out = processor(
            text=[sample["prompt"]],
            images=[padded_image],
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        proc_outs.append(proc_out)
        transformed_bboxes_list.append(torch.tensor(transformed_bbox, dtype=torch.float32))
        padded_images_for_opv.append(padded_image)
        
    # --- Step D: Collate all processed data into final tensors ---
   
    ids_list  = [o.input_ids[0] for o in proc_outs]
    mask_list = [o.attention_mask[0] for o in proc_outs]
    text_padded = tokenizer.pad(
        {"input_ids": ids_list, "attention_mask": mask_list},
        padding="longest",
        return_tensors="pt",
    )
    B, L = text_padded["input_ids"].shape

    # Collate image features
    pv_list = [o["pixel_values"] for o in proc_outs]
    pixel_values_flat = torch.cat(pv_list, dim=0)
    
    if pixel_values_flat.dim() > 2: # Handle different processor outputs
        pixel_values = pixel_values_flat
    else:
        pixel_values = pixel_values_flat.view(batch_size, -1, pixel_values_flat.shape[-1])

    # Collate original pixel values (which are now padded)
    original_pixel_values = torch.stack(
        [T.PILToTensor()(img) for img in padded_images_for_opv]
    ).permute(0, 2, 3, 1)

    # Create other tensors
    position_ids = torch.arange(L, dtype=torch.long, device=text_padded["input_ids"].device).unsqueeze(0).repeat(B, 1)
    raw_prompt_ids = [tokenizer.encode(b["prompt"], add_special_tokens=False) for b in batch]
    
    # Assemble the final dictionary
    out = {
        "input_ids":             text_padded["input_ids"],
        "attention_mask":        text_padded["attention_mask"],
        "position_ids":          position_ids,
        "pixel_values":          pixel_values,
        "original_pixel_values": original_pixel_values, 
        "gt_bboxes_xywh":        torch.stack(transformed_bboxes_list),
        "image_dims":            torch.tensor([list(target_size)] * B, dtype=torch.int32),
        "trajectory_ids":        torch.tensor([b["trajectory_ids"] for b in batch], dtype=torch.long),
        "turn_ids":              torch.tensor([b["turn_ids"] for b in batch], dtype=torch.long),
        "original_prompts":      np.array([b["prompt"] for b in batch], dtype=object),
        "raw_prompt_ids":        np.array(raw_prompt_ids, dtype=object),
    }
    
    return out


def create_mgrpo_v_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> tuple:
    """
    Creates training and validation dataloaders for MGRPO-V.
    """
    train_dataset = MGRPODataset(
        data_path=config.train_files,
        image_dir=config.image_dir,
        tokenizer=tokenizer,
        processor=processor,
    )
    
    if config.shuffle:
        sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(config.seed))
    else:
        sampler = SequentialSampler(train_dataset)

    train_batch_size = config.mini_rollout_batch_size or config.rollout_batch_size

    collate_wrapper = partial(mgrpo_v_collate_fn, processor=processor, tokenizer=tokenizer)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=config.dataloader_num_workers or 8,
        collate_fn=collate_wrapper,
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = MGRPODataset(
        data_path=config.val_files,
        image_dir=config.image_dir,
        tokenizer=tokenizer,
        processor=processor,
    )

    val_batch_size = config.val_batch_size if config.val_batch_size != -1 else len(val_dataset)

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers or 8,
        collate_fn=collate_wrapper,
        pin_memory=True,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1, "Training dataloader is empty!"
    assert len(val_dataloader) >= 1, "Validation dataloader is empty!"
    logging.info(f"Size of MGRPO-V train dataloader: {len(train_dataloader)}")
    logging.info(f"Size of MGRPO-V val dataloader: {len(val_dataloader)}")
    
    return train_dataloader, val_dataloader

def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        image_dir=config.image_dir,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
