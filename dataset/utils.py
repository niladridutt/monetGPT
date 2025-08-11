"""
Utility functions for dataset generation.
"""
import base64
import random
import numpy as np
import yaml
import io
import time
from .constants import HSV_PREDEFINED_COLORS, COLOR_OPERATIONS_MAPPING
import cv2


def load_config(config_path="configs/dataset_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def tif_to_png_encoded(file_path):
    """Convert TIF image to PNG and encode as base64."""
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for this function")
        
    from PIL import Image
    with Image.open(file_path) as img:
        with io.BytesIO() as buffer:
            img.save(buffer, format="PNG")
            png_data = buffer.getvalue()
    return base64.b64encode(png_data).decode('utf-8')


def calculate_color_percentages_hsv(image_path, hsv_color_dict=None):
    """
    Calculate color percentages using predefined HSV ranges.
    Args:
        image_path (str): Path to the image file.
        hsv_color_dict (dict): Dictionary mapping color name -> list of (lower, upper) HSV tuples.
    Returns:
        dict: color -> percentage of pixels in that color range
    """
    
    if hsv_color_dict is None:
        hsv_color_dict = HSV_PREDEFINED_COLORS
        
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
        
    height, width = image_bgr.shape[:2]
    scale = min(600 / width, 600 / height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    total_pixels = image_hsv.shape[0] * image_hsv.shape[1]
    
    color_counts = {}
    for color_name, hsv_ranges in hsv_color_dict.items():
        combined_mask = np.zeros((image_hsv.shape[0], image_hsv.shape[1]), dtype=np.uint8)
        
        for (lower, upper) in hsv_ranges:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(image_hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        color_count = np.sum(combined_mask > 0)
        color_counts[color_name] = color_count
    
    color_percentages = {
        color: (count / total_pixels) * 100.0
        for color, count in color_counts.items()
    }
    
    return color_percentages


def process_image_for_colors(image_path, threshold=0.04):
    """Process image and return available operations based on dominant colors."""
    color_percentages = calculate_color_percentages_hsv(image_path)
    total = sum(color_percentages.values())

    sorted_colors = sorted(
        (
            (color, pct / total)
            for color, pct in color_percentages.items()
            if (pct / total) > threshold
        ),
        key=lambda x: x[1],
        reverse=True
    )

    if sorted_colors:
        top_colors = sorted_colors[:5]
        selected_color = random.choice([color for color, _ in top_colors])
        return COLOR_OPERATIONS_MAPPING.get(selected_color, [])
    else:
        return []


def create_openai_client(config):
    """Create OpenAI client with configuration."""
    from openai import OpenAI
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )


def send_request_with_retry(request_func, *args, retries=None, delay=None, config=None):
    """Send request with retry logic on TypeError."""
    if config:
        retries = retries or config.get("retry_attempts", 1)
        delay = delay or config.get("timeout", 60)
    
    try:
        return request_func(*args)
    except TypeError:
        return "TypeError"
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return ""


def generate_value_normal_distribution(sigma=25):
    """Generate value using normal distribution centered at 0."""
    mu = 0
    size = 1
    random_floats = np.random.normal(mu, sigma, size)
    random_integers = np.clip(np.round(random_floats), -100, 100)
    return int(random_integers[0])


def generate_values_with_pattern(base_range, increment_range, value3_range):
    """Generate values following specific pattern for puzzle 2."""
    value1 = random.choice(base_range)
    value2_increment = random.choice(increment_range)
    value2 = value1 + value2_increment
    value3 = random.choice(value3_range) * -1
    
    values = [value1, value2, value3]
    
    if random.choice([True, False]):
        values = [-v for v in values]
    
    return values
