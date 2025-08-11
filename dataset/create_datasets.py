"""
Data formatting functions for creating ShareGPT format datasets.
"""
import json
import glob
import os
from .utils import load_config


def create_sharegpt_format_puzzle1(user_prompt, reasoning, image_path=None):
    """Create ShareGPT format entry for puzzle 1."""
    entry = {
        "messages": [
            {
                "content": "You are a helpful advanced image-editing assistant with expertise in Adobe Lightroom.",
                "role": "system"
            },
            {
                "content": user_prompt,
                "role": "user"
            },
            {
                "content": reasoning,
                "role": "assistant"
            }
        ]
    }
    if image_path:
        entry["images"] = [image_path]
    return entry


def create_sharegpt_format_puzzle2(operation, question_2, answer_2, reasoning, image_path=None):
    """Create ShareGPT format entry for puzzle 2."""
    template_prompt = f"""
<image> 
I have a stitched image of 4 Lightroom edits of the original photo where the `{operation}` has been adjusted to create 4 versions: a, b, c, and d (from left to right). These are in a random order of adjustment values.

Based on this stitched image, I need you to:

1. **Sort the images** from lowest to highest `{operation}` adjustment and provide the order.
2. **Identify the optimal image** and explain why it has the best level of `{operation}` for this specific photo.

Use detailed visual analysis and provide reasoning for both the sorting and the optimal choice.
"""

    entry = {
        "messages": [
            {
                "content": "You are a helpful advanced image-editing assistant with expertise in Adobe Lightroom.",
                "role": "system"
            },
            {
                "content": template_prompt,
                "role": "user"
            },
            {
                "content": reasoning,
                "role": "assistant"
            },
            {
                "content": question_2,
                "role": "user"
            },
            {
                "content": answer_2,
                "role": "assistant"
            }
        ]
    }
    if image_path:
        entry["images"] = [image_path]
    return entry


def create_sharegpt_format_puzzle3(operation, adjustment_values, reasoning, image_path):
    """Create ShareGPT format entry for puzzle 3."""
    adjustment_values = f"```json\n{json.dumps(adjustment_values, indent=2)}\n```"
    short_operation = operation.split("\n")[0]
    
    template_prompt = f"""
<image> 

Analyze the provided image and develop a professional-grade editing plan using operations available in Adobe Lightroom to adress issues in {operation}. 

Your task is to identify all visual issues in the image and propose precise, optimized adjustments to address issues in {short_operation}. 

Create a professional editing plan for this photo with a list of the **optimal** adjustments needed to address the identified issues.

For each adjustment, follow this format:
Adjustment: [Mention the adjustment that needs to be made. Eg: **Adjustment:** The whites need to be greatly reduced**.]
Issue: [Explain the specific issue in the image, focusing on how it negatively impacts the photo, use as much context from the image as possible.]
Solution: [Describe how the adjustment will resolve the issue]

You should ensure that applying these adjustments will lead to an **optimal** image with balanced adjustment values specifically tuned for this image.
"""
    
    question_2 = f"""
Based on the editing plan and the original image, tell the **optimal** adjustment values needed to edit this photo in JSON format. All adjustment values are scaled between -100 and +100. You must ensure that the final edited image has **optimal** adjustment values to look like an **optimal image**.

The following legend can be used to map the intensity values from your previous answer:
1-12: Very Slight
13-24: Slight
25-36: Mild
37-48: Moderate
49-60: Noticeable
61-72: Significant
73-84: Very Significant
85-100: Extremely Intense
"""

    answer_2 = f"""
Applying the below adjustment values in Lightroom will make the image **optimal**.

{adjustment_values}
"""

    entry = {
        "messages": [
            {
                "content": "You are a helpful advanced image-editing assistant with expertise in Adobe Lightroom.",
                "role": "system"
            },
            {
                "content": template_prompt,
                "role": "user"
            },
            {
                "content": reasoning,
                "role": "assistant"
            },
            {
                "content": question_2,
                "role": "user"
            },
            {
                "content": answer_2,
                "role": "assistant"
            }
        ]
    }
    if image_path:
        entry["images"] = [image_path]
    return entry


def create_dataset_puzzle1(config_path="configs/dataset_config.yaml"):
    """Create dataset for puzzle 1."""
    config = load_config(config_path)
    puzzle_config = config["puzzles"]["puzzle1"]
    
    user_prompt = """
<image> 
Here's a stitched pair of images (original and edited). Analyze the changes and provide the following:
Question: An image editing operation has applied in Lightroom. The value ranges from -100 to +100.

First, identify the operation by analyzing the change. Second, idenitify the adjustment value. Justify with reasoning.

Follow this template to answer your question.
1. The editing operation(s) applied.
2. The value of each adjustment.
3. Your step-by-step reasoning.


Use the following legend to describe the degree of adjustment:
1-12: Very Slight
13-24: Slight
25-36: Mild
37-48: Moderate
49-60: Noticeable
61-72: Significant
73-84: Very Significant
85-100: Extremely Intense


List of operations available: [
    "Blacks",
    "BlueHue",
    "BlueSaturation",
    "Clarity",
    "Contrast",
    "Exposure",
    "GreenHue",
    "GreenSaturation",
    "Highlights",
    "HueAdjustmentAqua",
    "HueAdjustmentBlue",
    "HueAdjustmentGreen",
    "HueAdjustmentYellow",
    "HueAdjustmentRed",
    "HueAdjustmentMagenta",
    "HueAdjustmentOrange",
    "HueAdjustmentPurple",
    "LuminanceAdjustmentAqua",
    "LuminanceAdjustmentBlue",
    "LuminanceAdjustmentGreen",
    "LuminanceAdjustmentYellow",
    "LuminanceAdjustmentRed",
    "LuminanceAdjustmentOrange",
    "LuminanceAdjustmentMagenta",
    "LuminanceAdjustmentPurple",
    "RedHue",
    "RedSaturation",
    "Saturation",
    "SaturationAdjustmentAqua",
    "SaturationAdjustmentBlue",
    "SaturationAdjustmentGreen",
    "SaturationAdjustmentYellow",
    "SaturationAdjustmentMagenta",
    "SaturationAdjustmentOrange",
    "SaturationAdjustmentPurple",
    "SaturationAdjustmentRed",
    "Shadows",
    "Sharpness",
    "Temperature",
    "Tint",
    "Vibrance",
    "VignetteAmount",
    "Whites"
]
----

Example:
1. Operation Applied: Saturation Adjustment.
2. Value of Adjustment: +25 saturation.

Reasoning:
Step 1: Observed that the colors in the edited image appear more vibrant compared to the original image.
Step 2: Identified that this increase in vibrancy is uniform across all colors, pointing to a saturation adjustment rather than a change in individual color channels.
Step 3: Compared the vibrancy to reference images with known saturation adjustments, approximating the change to +25.

Summary:
The operation applied was a 'Saturation Adjustment', with an approximate value of +25.
"""

    messages = []
    reasoning_files_path = puzzle_config["reasoning_path"]
    images_path_pattern = puzzle_config["images_path"]
    train_files = glob.glob(reasoning_files_path)
    
    # Extract extension from config pattern (e.g., "*.png" -> ".png")
    image_ext = images_path_pattern.split("*")[-1]

    for reasoning_file_path in train_files:
        # Replace reasoning path with images path and use config extension
        image_path = reasoning_file_path.replace("reasoning", "images").replace(".txt", image_ext)
        if not os.path.exists(image_path):
            reasoning_filename = os.path.basename(reasoning_file_path)
            print("image not found for", reasoning_filename)
            continue
            
        with open(reasoning_file_path, 'r', encoding="utf-8") as file:
            reasoning = file.read()
            
        if reasoning == "TypeError":
            print("TypeError", reasoning_file_path)
            continue
            
        if len(reasoning) < 3:
            print("reasoning too short", reasoning_file_path)
            continue

        reasoning = reasoning.replace("Operation(s)", "Operation")
        data_entry = create_sharegpt_format_puzzle1(user_prompt, reasoning, image_path)
        messages.append(data_entry)

    output_file = puzzle_config["output_file"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(messages, indent=4, ensure_ascii=False))
        f.write("\n")

    print(f"Data written to {output_file}")
    return messages


def create_dataset_puzzle2(config_path="configs/dataset_config.yaml"):
    """Create dataset for puzzle 2."""
    config = load_config(config_path)
    puzzle_config = config["puzzles"]["puzzle2"]
    
    messages = []
    reasoning_files_path = puzzle_config["reasoning_path"]
    images_path_pattern = puzzle_config["images_path"]
    train_files = glob.glob(reasoning_files_path)
    
    # Extract extension from config pattern (e.g., "*.png" -> ".png")
    image_ext = images_path_pattern.split("*")[-1]

    for reasoning_file_path in train_files:
        filename = reasoning_file_path.split("/")[-1].split(".")[0]
        parts = filename.split("_")
        operation = parts[2]
        order = "_".join(parts[3:])
        
        # Replace reasoning path with images path and use config extension
        image_path = reasoning_file_path.replace("reasoning", "images").replace(".txt", image_ext)
        if not os.path.exists(image_path):
            print("image not found for", filename)
            continue
            
        with open(reasoning_file_path, 'r', encoding="utf-8") as file:
            reasoning = file.read()
            
        if reasoning == "TypeError":
            print("TypeError", reasoning_file_path)
            continue
            
        if len(reasoning) < 3:
            print("reasoning too short", reasoning_file_path)
            continue

        question_2 = f"""Based on this analysis, what are the 3 adjustment values that were used to create versions a, b, and c? Provide them as a JSON list in the same order as the images [a, b, c]."""
        
        answer_2 = f"""```json\n{order.replace('_', ', ').replace('[', '').replace(']', '').split(', ')}\n```"""
        
        data_entry = create_sharegpt_format_puzzle2(operation, question_2, answer_2, reasoning, image_path)
        messages.append(data_entry)

    output_file = puzzle_config["output_file"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(messages, indent=4, ensure_ascii=False))
        f.write("\n")

    print(f"Data written to {output_file}")
    return messages


def create_dataset_puzzle3(config_path="configs/dataset_config.yaml"):
    """Create dataset for puzzle 3."""
    config = load_config(config_path)
    puzzle_config = config["puzzles"]["puzzle3"]
    generation_config = config["generation"]
    
    from .constants import OPERATION_MAP
    
    messages = []
    reasoning_files_path = puzzle_config["reasoning_path"]
    images_path_pattern = puzzle_config["images_path"]
    train_files = glob.glob(reasoning_files_path)
    
    # Get base paths from config
    images_base_path = puzzle_config["images_base_path"]
    configs_base_path = generation_config["output_dirs"]["puzzle3"]
    
    # Extract extension from config pattern (e.g., "*/*.tif" -> ".tif")
    image_ext = images_path_pattern.split("*")[-1]
    
    data_entry_count = 0
    skip = 0
    repeat = 0
    ops = ["white-balance-tone-contrast", "color-temperature", "hsl"]

    for reasoning_file_path in train_files:
        filename = reasoning_file_path.split("/")[-1].split(".")[0]
        
        operation = ""
        for op in ops:
            if op in filename:
                operation = op
                break
        
        operation_details = OPERATION_MAP[operation]
        
        # Construct image path using config extension and operation subdirectory
        image_path = f"{images_base_path}/{operation}/{filename}{image_ext}"

        if not os.path.exists(image_path):
            print("image not found for", filename)
            continue
            
        with open(reasoning_file_path, 'r', encoding="utf-8") as file:
            reasoning = file.read()
            
        if reasoning == "TypeError":
            print("TypeError", reasoning_file_path)
            continue
            
        if len(reasoning) < 3:
            print("reasoning too short", reasoning_file_path)
            continue

        configs_path = f"{configs_base_path}/{filename}.json"
        with open(configs_path, 'r') as file:
            adjustment_values = json.load(file)

        data_entry = create_sharegpt_format_puzzle3(operation_details, adjustment_values, reasoning, image_path)
        messages.append(data_entry)
        data_entry_count += 1

    output_file = puzzle_config["output_file"]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(messages, indent=4, ensure_ascii=False))
        f.write("\n")

    print(f"Data written to {output_file}")
    print("data_entry_count", data_entry_count)
    print("skip", skip)
    print("repeat", repeat)
    return messages


def create_all_datasets(config_path="configs/dataset_config.yaml"):
    """Create all puzzle datasets."""
    print("Creating puzzle 1 dataset...")
    create_dataset_puzzle1(config_path)
    
    print("Creating puzzle 2 dataset...")
    create_dataset_puzzle2(config_path)
    
    print("Creating puzzle 3 dataset...")
    create_dataset_puzzle3(config_path)
    
    print("All datasets created successfully!")
