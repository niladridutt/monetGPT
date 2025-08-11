"""
Core editing pipeline that combines GIMP and non-GIMP operations.
"""
import os
from typing import Optional
from .utils import (
    load_combined_config, 
    update_pipeline_file_paths, 
    execute_gimp_pipeline,
    ensure_directory
)
from image_ops.non_gimp_ops import execute_non_gimp_pipeline as _execute_non_gimp


def execute_non_gimp_pipeline(config_path: str, src_path: str, output_path: str):
    """Import and execute non-GIMP pipeline."""
    # Import here to avoid circular imports
    import sys
    sys.path.append('.')
    
    # Ensure output directory exists
    ensure_directory(os.path.dirname(output_path))
    
    return _execute_non_gimp(config_path, src_path, output_path)


class ImageEditingPipeline:
    """Main pipeline for applying image editing operations."""
    
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config = load_combined_config(config_path)
        self.pipeline_file = self.config["gimp"]["pipeline_file"]
    
    def execute_edit(
        self, 
        config_path: str, 
        src_image_path: str, 
        output_path: str
    ) -> None:
        """Execute complete editing pipeline (non-GIMP + GIMP operations)."""
        # First run non-GIMP operations
        execute_non_gimp_pipeline(config_path, src_image_path, output_path)
        
        # Then run GIMP operations on the result
        update_pipeline_file_paths(
            self.pipeline_file, 
            config_path, 
            output_path, 
            output_path
        )
        execute_gimp_pipeline(self.config)
    
    def execute_gimp_only(
        self,
        config_path: str,
        src_image_path: str, 
        output_path: str
    ) -> None:
        """Execute only GIMP operations."""
        update_pipeline_file_paths(
            self.pipeline_file,
            config_path,
            src_image_path,
            output_path
        )
        execute_gimp_pipeline(self.config)
    
    def execute_non_gimp_only(
        self,
        config_path: str,
        src_image_path: str,
        output_path: str
    ) -> None:
        """Execute only non-GIMP operations."""
        execute_non_gimp_pipeline(config_path, src_image_path, output_path)
