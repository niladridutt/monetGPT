"""
Puzzle generation for creating training data.
"""
import json
import random
import os
import glob
from .utils import (
    load_config, 
    process_image_for_colors, 
    generate_value_normal_distribution,
    generate_values_with_pattern
)
from .constants import (
    STANDARD_OPERATIONS, 
    WHITE_BALANCE_TONE_CONTRAST, 
    COLOR_TEMPERATURE,
    COLOR_SPECIFIC_ADJUSTMENTS
)
from tqdm import tqdm


def generate_trials_puzzle1(parameter_list, num_trials, prefix, output_dir):
    """Generate JSON files for puzzle 1."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for _ in range(num_trials):
        parameter = random.choice(parameter_list)
        value = random.randint(-100, 100)
        data = {parameter: value}
        file_name = f"{prefix}_{parameter}_{value}.json"
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)


def generate_trials_puzzle2(parameter_list, num_trials, prefix, output_dir):
    """Generate JSON files for puzzle 2."""
    parameter = random.choice(parameter_list)
    sub_folder = os.path.join(output_dir, f"{prefix}_{parameter}")
    os.makedirs(sub_folder, exist_ok=True)
    
    base_range = list(range(-75, -35)) + list(range(35, 76))
    increment_range = list(range(10, 26))
    value3_range = list(range(15, 46))
    
    values = generate_values_with_pattern(base_range, increment_range, value3_range)
    order = ['a', 'b', 'c']
    random.shuffle(order)
    
    for i, (label, value) in enumerate(zip(order, values)):
        data = {parameter: value}
        file_name = f"{label}.json"
        file_path = os.path.join(sub_folder, file_name)
        
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    
    return parameter


def generate_trials_puzzle3(operations_1, operations_2, available_color_operations, filename, output_dir):
    """Generate JSON files for puzzle 3."""
    small_parameters = ["Temperature", "Exposure", "Clarity", "Sharpness", "Tint"]
    data = {}
    operations = [operations_1, operations_2, available_color_operations]
    weights = [0.3, 0.25, 0.45]
    
    if len(available_color_operations) == 0:
        weights = [0.55, 0.45, 0.0]
    
    title = ["white-balance-tone-contrast", "color-temperature", "hsl"]
    indexes = list(range(len(operations)))
    choice = random.choices(indexes, weights=weights, k=1)[0]
    operations = operations[choice]
    
    num_operations = random.choices([2, 3, 4], weights=[0.5, 0.35, 0.15], k=1)[0]
    if len(operations) < num_operations:
        num_operations = len(operations)
    
    selected_operations = random.sample(operations, num_operations)
    
    for op in selected_operations:
        if op in small_parameters:
            value = generate_value_normal_distribution(sigma=15)
        else:
            value = generate_value_normal_distribution(sigma=25)
        data[op] = value
    
    file_name = f"{filename}_{title[choice]}.json"
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    return title[choice]


def generate_puzzle1(config_path="configs/dataset_config.yaml"):
    """Generate puzzles for puzzle 1."""
    config = load_config(config_path)
    generation_config = config["generation"]
    
    output_dir = generation_config["output_dirs"]["puzzle1"]
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = sorted(glob.glob(generation_config["image_sources"]["ppr10k_source"]))
    
    if not image_paths:
        print(f"Warning: No images found in {generation_config['image_sources']['ppr10k_source']}. Skipping puzzle 1 generation.")
        return
    
    for image_path in tqdm(image_paths, desc="Generating Puzzle 1"):
        try:
            available_color_operations = process_image_for_colors(image_path)
            prefix = image_path.split("/")[-1].split(".")[0]
            
            generate_trials_puzzle1(
                STANDARD_OPERATIONS, 
                generation_config["num_standard_trials"], 
                prefix, 
                output_dir
            )
            
            generate_trials_puzzle1(
                available_color_operations, 
                generation_config["num_color_trials"], 
                prefix, 
                output_dir
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def generate_puzzle2(config_path="configs/dataset_config.yaml"):
    """Generate puzzles for puzzle 2."""
    config = load_config(config_path)
    generation_config = config["generation"]
    
    output_dir = generation_config["output_dirs"]["puzzle2"]
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(generation_config["image_sources"]["ppr10k_target"])
    filtered_paths = []
    
    for image_path in image_paths:
        prefix = image_path.split("/")[-1].split(".")[0]
        if len(prefix) <= 4:
            filtered_paths.append(image_path)
    
    if not filtered_paths:
        print(f"Warning: No images found in {generation_config['image_sources']['ppr10k_target']}. Skipping puzzle 2 generation.")
        return
    
    random.shuffle(filtered_paths)
    
    for image_path in tqdm(filtered_paths, desc="Generating Puzzle 2"):
        try:
            available_color_operations = process_image_for_colors(image_path, threshold=0.06)
            prefix = image_path.split("/")[-1].split(".")[0]
            
            operations = list(STANDARD_OPERATIONS)
            
            op = generate_trials_puzzle2(operations, 1, prefix, output_dir)
            operations.remove(op)
            generate_trials_puzzle2(operations, 1, prefix, output_dir)
            
            generate_trials_puzzle2(available_color_operations, 1, prefix, output_dir)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def generate_puzzle3(config_path="configs/dataset_config.yaml"):
    """Generate puzzles for puzzle 3."""
    config = load_config(config_path)
    generation_config = config["generation"]
    
    output_dir = generation_config["output_dirs"]["puzzle3"]
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = sorted(glob.glob(generation_config["image_sources"]["ppr10k_target"]))
    filtered_paths = []
    
    for image_path in image_paths:
        prefix = image_path.split("/")[-1].split(".")[0]
        if len(prefix) <= 4:
            filtered_paths.append(image_path)

    num_trials = config["generation"]["num_puzzle3_trials"]
    
    # Check if we have enough images
    if not filtered_paths:
        print(f"Warning: No images found in {generation_config['image_sources']['ppr10k_target']}. Skipping puzzle 3 generation.")
        return
    
    # If we have fewer images than requested trials, use all available
    if len(filtered_paths) < num_trials:
        print(f"Warning: Only {len(filtered_paths)} images available, but {num_trials} trials requested. Using all available images.")
        num_trials = len(filtered_paths)
    
    filtered_paths = random.choices(filtered_paths, k=num_trials)
    random.shuffle(filtered_paths)
    
    counts = [0, 0, 0]
    
    for image_path in tqdm(filtered_paths, desc="Generating Puzzle 3"):
        try:
            available_color_operations = process_image_for_colors(image_path, threshold=0.035)
            prefix = image_path.split("/")[-1].split(".")[0]
            
            # Add random prefix for style variation
            style_prefix = random.choice(["00", "01", "02", "03", "04"])
            filename = f"{style_prefix}_{prefix}"
            
            operation_type = generate_trials_puzzle3(
                WHITE_BALANCE_TONE_CONTRAST,
                COLOR_TEMPERATURE,
                available_color_operations,
                filename,
                output_dir
            )
            
            if operation_type == "white-balance-tone-contrast":
                counts[0] += 1
            elif operation_type == "color-temperature":
                counts[1] += 1
            elif operation_type == "hsl":
                counts[2] += 1
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    print("Generation stats:", counts)


def generate_all_puzzles(config_path="configs/dataset_config.yaml"):
    """Generate all puzzles."""
    print("Generating puzzle 1...")
    generate_puzzle1(config_path)
    
    print("Generating puzzle 2...")
    generate_puzzle2(config_path)
    
    print("Generating puzzle 3...")
    generate_puzzle3(config_path)
    
    print("All puzzles generated successfully!")
