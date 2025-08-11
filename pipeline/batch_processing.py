"""
Batch processing for model predictions and comparisons.
"""
import glob
import os
import sys
from typing import List, Optional
from .core import ImageEditingPipeline
from .utils import load_combined_config, ensure_directory
from .stitch import merge_images_with_captions


class BatchProcessor:
    """Handles batch processing of predictions and comparisons."""
    
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config = load_combined_config(config_path)
        self.pipeline = ImageEditingPipeline(config_path)
    
    def execute_batch_predictions(self, target_editor: str = "") -> None:
        """Execute batch predictions from JSON configs."""
        print("Executing batch predictions...")
        
        paths = glob.glob("./ppr10k/qwen_direct/*.json")
        
        for idx, config_path in enumerate(paths):
            try:
                index = config_path.split("/")[-1].split(".")[0]
                print(f"Processing {index}")
                
                src_image_path = f"./ppr10k/sampled_tests/{index}.tif"
                image_path = f"./ppr10k/export/{index}_{target_editor}_gimp.png"
                output_path = f"./ppr10k/export/{index}_{target_editor}_final.png"
                merge_path = f"./gallery_page/predictions/{idx + 1}.png"
                
                # Ensure output directories exist
                ensure_directory(os.path.dirname(image_path))
                ensure_directory(os.path.dirname(output_path))
                ensure_directory(os.path.dirname(merge_path))
                
                # Execute pipeline
                self.pipeline.execute_gimp_only(config_path, src_image_path, image_path)
                self.pipeline.execute_non_gimp_only(config_path, image_path, output_path)
                
                # Create comparison
                merge_images_with_captions(
                    [src_image_path, output_path],
                    ["Original", "Qwen direct"],
                    merge_path
                )
                
            except Exception as e:
                print(f"Error processing {config_path}: {e}")
    
    def execute_monetgpt_predictions(self, target_editor: str = "a") -> None:
        """Execute MonetGPT model predictions and comparisons."""
        print("Executing MonetGPT predictions...")
        
        paths = glob.glob("./ppr10k/preds/predicted_settings/*.json")
        
        for config_path_predicted in paths:
            try:
                index = config_path_predicted.split("/")[-1].split(".")[0][:-2]
                
                # Set up paths
                config_path_editor = f"./ppr10k/xmp/settings/{index}_{target_editor}.json"
                src_image_path = f"./ppr10k/full_dataset/source/{index}.tif"
                output_path_editor = f"./ppr10k/export/{index}_{target_editor}_final.png"
                output_path_predicted = f"./ppr10k/export/{index}_predicted_final.png"
                merge_path = f"./ppr10k/preds/merged/{index}.png"
                
                # Ensure output directories exist
                ensure_directory(os.path.dirname(output_path_editor))
                ensure_directory(os.path.dirname(output_path_predicted))
                ensure_directory(os.path.dirname(merge_path))
                
                # Execute edits for both predictions and expert editor
                self.pipeline.execute_edit(
                    config_path_predicted, 
                    src_image_path, 
                    output_path_predicted
                )
                self.pipeline.execute_edit(
                    config_path_editor, 
                    src_image_path, 
                    output_path_editor
                )
                
                # Create three-way comparison
                merge_images_with_captions(
                    [src_image_path, output_path_editor, output_path_predicted],
                    ["Original", "Expert Editor", "Predicted"],
                    merge_path,
                    text_file=config_path_predicted
                )
                
            except Exception as e:
                print(f"Error processing {config_path_predicted}: {e}")
    
    def process_single_config(
        self, 
        config_path: str, 
        src_image_path: str, 
        output_path: str
    ) -> None:
        """Process a single configuration file."""
        print(f"Processing single config: {config_path}")
        
        # Ensure output directory exists
        ensure_directory(os.path.dirname(output_path))
        
        self.pipeline.execute_edit(config_path, src_image_path, output_path)
