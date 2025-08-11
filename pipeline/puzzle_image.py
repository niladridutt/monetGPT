"""
Puzzle image generation for different puzzle types.
Creates visual puzzle images by applying JSON configurations to source images.
"""
import glob
import json
import os
import random
import sys
from typing import List, Dict, Optional

from .core import ImageEditingPipeline
from .stitch import merge_images_with_captions
from .utils import (
    load_combined_config, 
    load_pipeline_config,
    flip_json_signs, 
    ensure_directory
)


class PuzzleImageGenerator:
    """Generator for puzzle images across different puzzle types."""
    
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config = load_combined_config(config_path)
        self.pipeline = ImageEditingPipeline(config_path)
    
    def generate_puzzle1_images(self) -> None:
        """Generate puzzle images for puzzle 1 (single operation analysis)."""
        print("Generating Puzzle 1 images...")
        
        # Use puzzle-based paths from dataset config
        configs_path = f"{self.config['generation']['output_dirs']['puzzle1']}/*.json"
        images_output = f"{self.config['puzzles']['puzzle1']['images_base_path']}"
        
        # Extract source directory and extension from config
        source_pattern = self.config["image_sources"]["ppr10k_source"]
        source_base = source_pattern.split("*")[0]  # Remove glob pattern, get directory
        source_ext = source_pattern.split("*")[1] if "*" in source_pattern else ".png"  # Get extension
        
        ensure_directory(images_output)
        
        paths = glob.glob(configs_path)
        random.shuffle(paths)
        
        for idx, config_path in enumerate(paths):
            try:
                print(f"Processing {idx}: {config_path}")
                
                filename = config_path.split("/")[-1].split(".")[0]
                # Extract the base image name from filename (e.g., "image 33" from "image 33_Tint_-70")
                # Split by underscore, then take the part before the first underscore that contains the operation
                if "_" in filename:
                    # Find the operation part (starts after the image name)
                    parts = filename.split("_")
                    # First part should be the image name (e.g., "image 33")
                    image_index = parts[0]
                else:
                    image_index = filename
                src_image_path = f"{source_base}{image_index}{source_ext}"
                
                temp_output = f"{self.config['image_processing']['temp_paths'][0]}"
                merge_path = f"{images_output}/{filename}.png"
                
                # Update pipeline and execute
                self.pipeline.execute_gimp_only(config_path, src_image_path, temp_output)
                self.pipeline.execute_non_gimp_only(config_path, temp_output, temp_output)
                
                # Create before/after comparison
                merge_images_with_captions(
                    [src_image_path, temp_output],
                    ["Original", "Edited"],
                    merge_path
                )
                
            except Exception as e:
                print(f"Error processing {config_path}: {e}")
    
    def generate_puzzle2_images(self) -> None:
        """Generate puzzle images for puzzle 2 (multi-version comparison)."""
        print("Generating Puzzle 2 images...")
        
        # Use puzzle-based paths from dataset config
        configs_base = self.config['generation']['output_dirs']['puzzle2']
        images_output = self.config['puzzles']['puzzle2']['images_base_path']
        temp_paths = self.config["image_processing"]["temp_paths"][1:4]  # Use temp_1, temp_2, temp_3
        
        ensure_directory(images_output)
        
        dirs = os.listdir(configs_base)
        print(f"Total configs: {len(dirs)}")
        
        for idx, dir_name in enumerate(dirs):
            try:
                print(f"Processing {idx}: {dir_name}")
                
                config_paths = glob.glob(f"{configs_base}/{dir_name}/*.json")
                if len(config_paths) != 3:
                    print(f"Skipping {dir_name} - invalid config count")
                    continue
                
                # Extract image name from directory name
                # e.g., "ip2p_Tint" -> "ip2p"
                dir_parts = dir_name.split("_")
                image_index = dir_parts[0]  # Just take the first part: "ip2p", "mgie", "ours"
                
                # Use the same source pattern as puzzle 1
                source_pattern = self.config["image_sources"]["ppr10k_source"]
                source_base = source_pattern.split("*")[0]
                source_ext = source_pattern.split("*")[1] if "*" in source_pattern else ".png"
                src_image_path = f"{source_base}{image_index}{source_ext}"
                
                # Process each config and collect data
                adjusted_vals = []
                output_paths = []
                
                for idx2, config_path in enumerate(config_paths):
                    output_path = temp_paths[idx2]
                    # For puzzle 2, we need to read the config to get the adjustment value
                    # instead of parsing it from the filename
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Get the main adjustment value (assuming single operation per config)
                    # Skip zero values and get the first non-zero value
                    adjusted_val = 0
                    for key, value in config_data.items():
                        if key not in ['Saturation', 'Contrast', 'Exposure'] or value != 0:
                            adjusted_val = int(value) if isinstance(value, (int, float)) else 0
                            break
                    
                    adjusted_vals.append(adjusted_val)
                    output_paths.append(output_path)
                    
                    self.pipeline.execute_edit(config_path, src_image_path, output_path)
                
                # Add original image (value = 0)
                adjusted_vals.append(0)
                output_paths.append(src_image_path)
                
                # Create value-to-path mapping and shuffle
                vals_to_path = {val: path for val, path in zip(adjusted_vals, output_paths)}
                random.shuffle(output_paths)
                
                # Create filename based on sorted order
                filename_order = []
                for key in sorted(vals_to_path.keys()):
                    pos = output_paths.index(vals_to_path[key])
                    filename_order.append(chr(97 + pos))  # a, b, c, d
                
                optimal_pos = output_paths.index(vals_to_path[0])
                filename_order.append(chr(97 + optimal_pos))
                
                merge_path = f"{images_output}/{dir_name}_{'_'.join(filename_order)}.png"
                
                merge_images_with_captions(
                    output_paths,
                    ["a", "b", "c", "d"],
                    merge_path,
                    dpi=self.config["image_processing"]["default_dpi"]
                )
                
            except Exception as e:
                print(f"Error processing {dir_name}: {e}")
    
    def generate_puzzle3_images(self) -> None:
        """Generate puzzle images for puzzle 3 (comprehensive editing plans)."""
        print("Generating Puzzle 3 images...")
        
        # Use puzzle-based paths from dataset config
        configs_path = f"{self.config['generation']['output_dirs']['puzzle3']}/*.json"
        
        paths = glob.glob(configs_path)
        random.shuffle(paths)
        
        operations = ["white-balance-tone-contrast", "color-temperature", "hsl"]
        
        for idx, config_path in enumerate(paths):
            try:
                print(f"Processing {idx}: {config_path}")
                
                filename = config_path.split("/")[-1].split(".")[0]
                # For puzzle 3, extract the image name from the middle part
                parts = filename.split("_")
                if len(parts) >= 2:
                    # Skip the number prefix and get the image name
                    image_index = parts[1]  
                else:
                    image_index = parts[0]
                
                # Use the same source pattern as puzzle 1
                source_pattern = self.config["image_sources"]["ppr10k_source"]
                source_base = source_pattern.split("*")[0]
                source_ext = source_pattern.split("*")[1] if "*" in source_pattern else ".png"
                src_image_path = f"{source_base}{image_index}{source_ext}"
                
                operation = next((op for op in operations if filename.endswith(op)), "")
                if not operation:
                    continue
                
                # Set up output paths using puzzle-based structure
                base_images_path = self.config['puzzles']['puzzle3']['images_base_path']
                output_dir = f"{base_images_path}/{operation}"
                output_dir2 = f"{base_images_path}2/{operation}"
                
                ensure_directory(output_dir)
                ensure_directory(output_dir2)
                
                output_path = f"{output_dir}/{filename}.tif"
                merge_path = f"{output_dir2}/{filename}.png"
                
                if os.path.exists(merge_path):
                    continue
                
                # Execute editing pipeline
                self.pipeline.execute_edit(config_path, src_image_path, output_path)
                
                # Flip signs since the puzzle images need to be flipped
                flip_json_signs(config_path)
                
                # Create comparison image
                merge_images_with_captions(
                    [output_path, src_image_path], 
                    ["Original", "Edited"],
                    merge_path
                )
                
            except Exception as e:
                print(f"Error processing {config_path}: {e}")
    
    def generate_all_puzzles(self) -> None:
        """Generate puzzle images for all puzzle types."""
        self.generate_puzzle1_images()
        self.generate_puzzle2_images() 
        self.generate_puzzle3_images()
