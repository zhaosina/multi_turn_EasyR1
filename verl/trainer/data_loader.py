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

# mgrpo_v_dataloader.py

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

# Assuming DataConfig is in this path, adjust if necessary
from .config import DataConfig

# --- New MGRPO-V Dataset Class ---

class MGRPODataset(Dataset):
    """
    A specialized dataset for MGRPO-V training.
    It only loads raw data; all processing is now handled by the collate_fn.
    """
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        # Add image_size to ensure consistent shapes for stacking
        image_size: tuple = (448, 448),
        prompt_key: str = "prompt",
        image_key: str = "image",
        gt_bbox_key: str = "ground_truth_bbox",
        traj_id_key: str = "trajectory_id",
        turn_id_key: str = "turn_id",
        **kwargs, # Absorb other arguments
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
        """
        Retrieves a single data sample and resizes the image.
        """
        item = self.data[idx]

        image_path = os.path.join(self.image_dir, item[self.image_key])
        try:
            image = Image.open(image_path).convert("RGB")
            # Resize image to ensure all pixel_values tensors have the same shape
            image = image.resize(self.image_size)
            img_w, img_h = image.size
        except Exception as e:
            logging.error(f"Could not load or resize image {image_path}: {e}")
            return {} 

        prompt_text = item[self.prompt_key]
        gt_bbox_xywh = item.get(self.gt_bbox_key)
        trajectory_id = item.get(self.traj_id_key)
        turn_id = item.get(self.turn_id_key)

        if gt_bbox_xywh is None or trajectory_id is None or turn_id is None:
            logging.error(f"Missing required MGRPO-V metadata in item {idx}.")
            return {}

        return {
            "image": image,
            "prompt": prompt_text,
            "gt_bboxes_xywh": torch.tensor(gt_bbox_xywh, dtype=torch.float32),
            "image_dims": torch.tensor([img_w, img_h], dtype=torch.int32),
            "trajectory_ids": torch.tensor(trajectory_id, dtype=torch.long),
            "turn_ids": torch.tensor(turn_id, dtype=torch.long),
        }

# --- FINAL ROBUST Collate Function (Based on your implementation) ---

def mgrpo_v_collate_fn(
    batch: List[Dict[str, Any]],
    *,
    processor: ProcessorMixin,
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, Any]:
    """
    Collates data for MGRPO-V, ensuring `position_ids` is created and
    that all returned values are tensors or numpy arrays compatible with DataProto.
    """
    batch = [b for b in batch if b]
    if not batch:
        return {}

    # 1. Process each sample individually
    proc_outs = []
    for b in batch:
        proc_out = processor(
            text=[b["prompt"]],
            images=[b["image"]],
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        proc_outs.append(proc_out)

    # 2. Pad text-related tensors
    ids_list  = [o.input_ids[0] for o in proc_outs]
    mask_list = [o.attention_mask[0] for o in proc_outs]
    text_padded = tokenizer.pad(
        {"input_ids": ids_list, "attention_mask": mask_list},
        padding="longest",
        return_tensors="pt",
    )
    B, L = text_padded["input_ids"].shape

    # 3. Create position_ids
    position_ids = torch.arange(L, dtype=torch.long, device=text_padded["input_ids"].device)
    position_ids = position_ids.unsqueeze(0).repeat(B, 1)

    # 4. Stack image-related tensors
    pv_list, thw_list = [], []
    for o in proc_outs:
        pv = o["pixel_values"]
        if pv.ndim == 2:
            pv = pv.unsqueeze(0)
        pv_list.append(pv)
        if "image_grid_thw" in o:
            g = o["image_grid_thw"]
            if g.ndim == 1:
                g = g.unsqueeze(0)
            thw_list.append(g)
            
    pixel_values = torch.cat(pv_list, dim=0)
    if thw_list:
        image_grid_thw = torch.cat(thw_list, dim=0)

    # 5. Create raw_prompt_ids as a NumPy object array to be compatible with DataProto
    raw_prompt_ids = [tokenizer.encode(b["prompt"], add_special_tokens=False) for b in batch]
    
    # Assemble the final dictionary
    out = {
        # Tensors
        "input_ids":       text_padded["input_ids"],
        "attention_mask":  text_padded["attention_mask"],
        "position_ids":    position_ids,
        "pixel_values":    pixel_values,
        "gt_bboxes_xywh":  torch.stack([b["gt_bboxes_xywh"] for b in batch]),
        "image_dims":      torch.stack([b["image_dims"]     for b in batch]),
        "trajectory_ids":  torch.stack([b["trajectory_ids"] for b in batch]),
        "turn_ids":        torch.stack([b["turn_ids"]       for b in batch]),
        # Non-Tensor (must be a numpy array for DataProto)
        "raw_prompt_ids":  np.array(raw_prompt_ids, dtype=object),
    }
    if thw_list:
        out["image_grid_thw"] = image_grid_thw

    return out

# --- Dataloader Creation Function (Updated) ---

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
