"""
Image editing pipeline package.
"""
import sys

# Check Python version and import appropriate modules
if sys.version_info[0] >= 3:
    # Python 3+ with type hints
    from .core import ImageEditingPipeline
    from .puzzle_image import PuzzleImageGenerator
    from .batch_processing import BatchProcessor
    from .utils import load_pipeline_config, load_combined_config
    
    __all__ = [
        'ImageEditingPipeline',
        'PuzzleImageGenerator',
        'BatchProcessor',
        'load_pipeline_config',
        'load_combined_config'
    ]
else:
    # Python 2.7 - don't import anything to avoid compatibility issues
    # GIMP will use the standalone pipeline.py file instead
    __all__ = []
