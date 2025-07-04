# preprocess_screenspot.py

import os
import json
import argparse
import random
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_screenspot_data(
    data_dir: str,
    output_dir: str,
    train_split_ratio: float = 0.95,
    seed: int = 42
):
    """
    Loads the screenspot-v2 dataset, converts it into the MGRPO-V trajectory format,
    and splits it into training and validation sets.

    The script performs the following transformations:
    1.  Combines desktop, mobile, and web JSON files.
    2.  For each data point, it adds:
        - 'trajectory_id': A unique ID for each sample.
        - 'turn_id': Set to 0, as these are all initial turns.
    3.  Renames keys to match the expected format of MGRPODataset.
    4.  Splits the combined data into train and validation sets.
    5.  Saves the output as .jsonl files, ready for the MGRPO-V dataloader.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

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
                logging.info(f"Loaded {len(data)} samples from {file_name}")
                all_samples.extend(data)
        else:
            logging.warning(f"File not found, skipping: {file_path}")

    if not all_samples:
        logging.error("No data loaded. Please check your data directory.")
        return

    logging.info(f"Total samples combined: {len(all_samples)}")
    logging.info("Converting to MGRPO-V trajectory format...")

    processed_data = []
    for i, item in enumerate(tqdm(all_samples, desc="Processing samples")):
        # Ensure the required keys exist
        if "instruction" not in item or "bbox" not in item or "img_filename" not in item:
            logging.warning(f"Skipping sample {i} due to missing keys.")
            continue

        processed_item = {
            # --- MGRPO-V specific fields ---
            "trajectory_id": i,
            "turn_id": 0,  # This is the initial turn
            
            # --- Renamed/mapped fields for MGRPODataset ---
            "prompt": item["instruction"],
            "image": item["img_filename"],
            "ground_truth_bbox": item["bbox"], # The key matches what we expect: [x, y, w, h]
        }
        processed_data.append(processed_item)

    # Shuffle and split the data
    logging.info(f"Shuffling and splitting data ({train_split_ratio*100}% train / {(1-train_split_ratio)*100}% val)...")
    random.shuffle(processed_data)
    
    split_index = int(len(processed_data) * train_split_ratio)
    train_data = processed_data[:split_index]
    val_data = processed_data[split_index:]

    # Save to .jsonl files
    train_output_path = os.path.join(output_dir, "train.jsonl")
    val_output_path = os.path.join(output_dir, "val.jsonl")

    def save_as_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logging.info(f"Saved {len(data)} samples to {path}")

    save_as_jsonl(train_data, train_output_path)
    save_as_jsonl(val_data, val_output_path)
    
    logging.info("Preprocessing complete!")
    logging.info(f"Train set: {train_output_path}")
    logging.info(f"Validation set: {val_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess screenspot-v2 data for MGRPO-V training.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to the root directory of the screenspot-v2 dataset (containing the JSON files)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the processed train.jsonl and val.jsonl files."
    )
    parser.add_argument(
        "--train_split_ratio", 
        type=float, 
        default=0.95, 
        help="The ratio of data to be used for the training set."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for shuffling and splitting the data."
    )
    args = parser.parse_args()
    
    preprocess_screenspot_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_split_ratio=args.train_split_ratio,
        seed=args.seed
    )