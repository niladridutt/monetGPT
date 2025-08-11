"""
Image Operations Module

This module contains all image processing operations used by MonetGPT:
- gimp_ops.py: GIMP-based image operations (brightness, contrast, saturation, etc.)
- non_gimp_ops.py: Non-GIMP operations and pipeline execution
- gimp_pipeline.py: GIMP pipeline configuration and execution logic
- image_dehazer/: Image dehazing operations

Environment Compatibility:
- Regular Python: non_gimp_ops work (requires PIL, skimage, cv2, etc.)
- GIMP Python: gimp_ops work (requires gimpfu, limited stdlib)
- Neither environment has access to both sets of operations

Usage:
    # For non-GIMP operations (regular Python environment)
    from image_ops import execute_non_gimp_pipeline
    
    # For GIMP operations (GIMP Python environment)
    from image_ops.gimp_ops import adjust_brightness, adjust_contrast
"""

__all__ = []

try:
    from .non_gimp_ops import execute_non_gimp_pipeline
    __all__.append('execute_non_gimp_pipeline')
except ImportError:
    pass  # Expected when in GIMP environment or missing dependencies

# Try to import GIMP operations (requires gimpfu)
try:
    from .gimp_ops import *
    from .gimp_pipeline import *
except ImportError:
    pass  # Expected when not in GIMP environment
