import os
from PIL import Image
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm

def analyze_resolutions(image_dir: str):
    """
    Analyzes image resolutions for different platforms (mobile, desktop, web).

    :param image_dir: Path to the directory containing all images.
    """
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found at '{image_dir}'")
        return

    # 使用defaultdict方便地按平台分组
    resolutions = defaultdict(list)
    platforms = ["mobile", "pc", "web"]

    print(f"Scanning images in '{image_dir}'...")
    
    # 获取所有文件名，以便tqdm显示进度条
    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(all_files, desc="Analyzing images"):
        platform = None
        for p in platforms:
            if filename.lower().startswith(p):
                platform = p
                break
        
        if platform:
            try:
                with Image.open(os.path.join(image_dir, filename)) as img:
                    width, height = img.size
                    resolutions[platform].append((width, height))
            except Exception as e:
                print(f"Warning: Could not read or process image {filename}. Error: {e}")

    print("\n--- Resolution Analysis Results ---")

    for platform, res_list in resolutions.items():
        if not res_list:
            print(f"\nPlatform: {platform.capitalize()} - No images found.")
            continue

        res_array = np.array(res_list)
        widths = res_array[:, 0]
        heights = res_array[:, 1]
        
        aspect_ratios = widths / heights

        print(f"\nPlatform: {platform.capitalize()} ({len(res_list)} images)")
        print("="*40)
        print(f"  Width  -- Min: {np.min(widths):>5}, Max: {np.max(widths):>5}, Mean: {np.mean(widths):>7.1f}, Std: {np.std(widths):>7.1f}")
        print(f"  Height -- Min: {np.min(heights):>5}, Max: {np.max(heights):>5}, Mean: {np.mean(heights):>7.1f}, Std: {np.std(heights):>7.1f}")
        print(f"  Aspect Ratio -- Min: {np.min(aspect_ratios):>5.2f}, Max: {np.max(aspect_ratios):>5.2f}, Mean: {np.mean(aspect_ratios):>5.2f}, Std: {np.std(aspect_ratios):>5.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze image resolutions from the ScreenSpotV2 dataset.")
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory with all platform images.')
    args = parser.parse_args()
    
    analyze_resolutions(args.image_dir)