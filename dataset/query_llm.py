"""
LLM query functions for generating reasoning explanations.
"""
import json
import glob
import os
import argparse
from tqdm import tqdm
from .utils import load_config, create_openai_client, encode_image, send_request_with_retry


def send_request_puzzle1(image_path, operation, value, config):
    """Send request for puzzle 1 reasoning generation."""
    client = create_openai_client(config)
    value = f"+{value}" if int(value) > 0 else f"{value}"
    
    original_image = encode_image(image_path)
    instruction = f"""
Here's a stitched pair of images (original and edited). Analyze the changes and provide the following:
Question: An expert image editor has applied the operation {operation} with value {value} in Lightroom. The value ranges from -100 to +100.
Justify with reasoning.

Follow this template to answer your question.
1. The editing operation(s) applied.
2. The value of each adjustment.
3. Your step-by-step reasoning.


Use the following legend to describe the degree of adjustment:
1–12: Very Slight
13–24: Slight
25–36: Mild
37–48: Moderate
49–60: Noticeable
61–72: Significant
73–84: Very Significant
85–100: Extremely Intense

Example:
1. Operation(s) Applied: Saturation Adjustment.
2. Value of Adjustment: +25% saturation.

Reasoning:
Step 1: Observed that the colors in the edited image appear more vibrant compared to the original image.
Step 2: Identified that this increase in vibrancy is uniform across all colors, pointing to a saturation adjustment rather than a change in individual color channels.
Step 3: Compared the vibrancy to reference images with known saturation adjustments, approximating the change to +25%.

Summary:
The operation applied was a 'Saturation Adjustment,' with an approximate value of +25%.
"""
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with expertise in image editing using Lightroom. You will compare the images side by side (images are stitched side by side so you can DEFINITELY do this)"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{original_image}",
                    },
                },
            ],
        }
    ]
    
    result = client.chat.completions.create(
        messages=messages, 
        model=config["model"]
    )
    return result.choices[0].message.content


def send_request_puzzle2(image_path, operation, order, config):
    """Send request for puzzle 2 reasoning generation."""
    client = create_openai_client(config)
    original_image = encode_image(image_path)
    
    index = order[-1]
    order = "[" + order[:-2] + "]"
    
    if not index.isalpha():
        index = order[-2]
    
    instruction = f"""
A source image has been edited in Lightroom where the `{operation}` has been adjusted to create 4 versions of the image: a, b, c, and d (from left to right and it's labelled as well). The stitched image represents these 4 versions in a random order of adjusted values.

Tasks:
1. **Sorting and Reasoning**:
   - The sorted order of the images from lowest to highest `{operation}` adjustment is: **`{order}`** (this is the answer).
   - Justify this order by explaining why each image has a lesser or greater level of `{operation}` compared to the others. When describing the reasoning:
     - Include as much context about the image as possible, such as how highlights, shadows, textures, colors, or other elements are affected by the adjustment.
     - Avoid generic or vague statements like "It balances the highlights." Instead, pinpoint exact issues caused by under- or over-adjustment (e.g., loss of detail in shadows, unnatural brightness, washed-out tones, etc.).

 **Image `{index}`** is the optimal image.

2. **Justification for optimal Image `{index}`**:
   - Explain why image **`{index}`** has the optimal level of `{operation}`.
   - Focus on how this level of adjustment specifically improves the image's visual quality while preserving critical details. Mention issues observed in other images and how the optimal level avoids those problems.
   - Address specific visual elements affected by the adjustment and why this level is optimal for maintaining balance and naturalness.

### Sample Response: (**You must follow the below structure/format**)

Sorted Ordering:
The sorted order of the images from lowest to highest Contrast adjustment is: c, a, b, d.

Justification:
Image c: This image has the least Contrast, as indicated by its very flat tonal range. The shadows appear washed out and lack depth, making the entire scene feel lifeless. For example, the tree bark in the foreground has almost no discernible texture, and the clouds in the sky are soft and undefined.
Image a: This image demonstrates moderate Contrast, with noticeable improvements. The tree bark reveals more texture, and the clouds in the sky show better definition. However, it still does not achieve the depth and vibrancy seen in b.
Image b: This image strikes the perfect balance of Contrast. The highlights in the clouds are bright but retain subtle textures, while the shadows in the tree bark are deep but still preserve intricate patterns and grooves. This level of Contrast enhances the image's tonal range without overemphasizing differences, creating a natural and visually appealing result.
Image d: This image has the highest level of Contrast, resulting in over-emphasized tonal differences. The shadows are overly dark, obscuring details in the tree bark and turning it into a uniform black mass. Similarly, the highlights in the clouds are blown out, losing subtle textures and making the sky appear unnaturally harsh.

Why Image b is optimal:
The operation applied is Contrast, and image b has the optimal level of adjustment.
This level of Contrast achieves a balanced tonal range that enhances visual quality while preserving critical details:
Shadows: The shadows are rich and deep, adding a sense of depth to the image, but they retain fine details, such as the individual grooves and cracks in the tree bark.
Highlights: The highlights are bright enough to make the clouds in the sky stand out, but they still maintain subtle gradients and textures, avoiding a washed-out or overly bright appearance.

Comparison to Other Images:
Compared to c, image b avoids the flat, lifeless appearance caused by low Contrast, where shadows and highlights lack differentiation.
Compared to d, image b avoids the harsh tonal extremes that result in loss of detail in both the highlights and shadows.
Image b is optimal because it reveals the intricate textures of the tree bark while maintaining the subtle gradients in the clouds, ensuring both shadow and highlight details are preserved. This balance enhances the scene's depth and vibrancy, creating a visually striking image that remains true to the natural appearance of the original scene.
"""
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant with expertise in image editing using Lightroom. You will analyze the stitched images and provide detailed reasoning."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{original_image}",
                    },
                },
            ],
        }
    ]
    
    result = client.chat.completions.create(
        messages=messages, 
        model=config["model"]
    )
    return result.choices[0].message.content


def send_request_puzzle3(image_path, settings, operation, extra, config):
    """Send request for puzzle 3 reasoning generation."""
    from .constants import SUMMARY_MAP
    
    client = create_openai_client(config)
    original_image = encode_image(image_path)
    short_operation = operation.split("\n")[0]

    instruction = f"""
Examine the side-by-side comparison image (original on the left, edited on the right) in detail to understand every single change the professional photographer made. 

We will be adressing issues in {operation}.

Identify all the issues in the original image that prompted these adjustments, being specific and descriptive (e.g., blown-out highlights, lack of contrast, color cast, mood issues, distracting background elements, etc.). Explain why each element might pose a problem or need adjusting.

{SUMMARY_MAP[extra]}

After identifying these issues, create a list of the adjustments needed to address them. Use this format for each adjustment:

[Start your answer by saying the category of operations it addresses. In this case: {short_operation}]

**Adjustment:** [Mention the adjustment that needs to be made, phrased as an instruction. E.g.: “The whites need to be greatly reduced."]  
**Issue:** [Explain the specific issue in the original image, focusing on how it negatively impacts the photo. Be explicit about the visual problem, referencing the original photo's elements—no vague statements or incomplete references.]  
**Solution:** [Describe how the adjustment will resolve the issue, phrased in a future-oriented tone. For instance, say “the edited image will have more balanced highlights" rather than “in the edited image, the highlights are balanced." **Make sure that the change is observable in the edited images**]

Add a very short summary at the end justifying how it meets the editing goal (balanced vs punchy colors) the expert editor was aiming for.

Include as much context as possible about the original and edited versions, referencing notable visual features that stand out or become altered in the final edit (such as lighting nuances, tonal ranges, color balance, focal elements, expressions, textures, or compositional details). 
Connect each edit to the specific elements in the original photo that need improving (for example, a color cast in the subject's shirt or underexposed shadows in the background) and describe precisely how the chosen adjustments resolve these issues.
**Make sure that the change is observable in the edited images**

The actual adjustments from the professional editor are given below. (All adjustments range from -100 to +100, where higher absolute values signify more intensity.)

Actual adjustments:
{settings}

These adjustments specifically address  operations for {operation}.

When describing the degree of each change, do not mention the numerical values themselves. Instead, use the following legend to characterize intensity:
1-12: Very Slight
13-24: Slight
25-36: Mild
37-48: Moderate
49-60: Noticeable
61-72: Significant
73-84: Very Significant
85-100: Extremely Intense

Avoid generic or vague statements like “It balances the highlights." Instead, clearly pinpoint the exact issue (e.g., overexposure in specific areas, color casting on the subject's face) and explain how the adjustment solves that problem. Also avoid uncertainty—base your justifications on the most probable reasons for each adjustment and reference the visible changes in the side-by-side comparison to confirm your reasoning. Additionally, do not explicitly refer to the present edited photo; use a future-oriented tone for the solution description (e.g., “the edited image will have deeper, more vibrant colors," rather than “in the edited image, the colors are deeper and more vibrant.").

Example of an adjustment:

**Adjustment:** The highlights need to be significantly reduced.  
**Issue:** In the original image, the highlights, particularly on the surface of the water and parts of the woman's dress, are too bright and verge on being blown out. These overexposed regions lack detail and create a visual distraction.  
**Solution:** By significantly reducing the highlights, the details in the brightest parts of the image will be recovered. The overexposed areas of the water will gain texture, the details in the dress will become more visible, and the overall image will have a more balanced exposure.

**Start your answer like below:
Editing Goal: {SUMMARY_MAP[extra]}**

Followed by [adjustments]

**Do not write any summary at the end and do not mention {short_operation} for every single adjustment, only mention it once at the beginning.**
"""

    messages = [
        {
            "role": "system",
            "content": """
You are an advanced image-editing assistant with expertise in Adobe Lightroom. You have the ability to analyze a stitched pair of images (original on the left, edited on the right). Your task is to produce a single set of instructions that identifies issues in the original image and provides justified adjustments to address them.

When describing your solutions, follow these rules:

1. **Do not refer to the present edited photo in a direct manner; instead, phrase any improvements in future tense (e.g., “the edited image will have...")**.
2. Provide detailed justifications for each suggested adjustment, linking them to the specific issues found in the original image.
3. Describe adjustment intensities using a descriptive legend (Very Slight, Slight, Mild, Moderate, etc.) instead of numerical values.
4. Avoid vague or generic statements—be precise about what is wrong in the original image and how the adjustment solves that issue.
5. Base your justifications on visible changes seen in the stitched comparison.
"""
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{original_image}",
                    },
                },
            ],
        }
    ]
    
    result = client.chat.completions.create(
        messages=messages, 
        model=config["model"]
    )
    return result.choices[0].message.content


def json_to_text(file_path):
    """Convert JSON config to text representation."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    text_parts = []
    for key, value in data.items():
        if value >= 0:
            text_parts.append(f"{key}: +{value}")
        else:
            text_parts.append(f"{key}: {value}")
    
    return ", ".join(text_parts)


def query_puzzle1(range_a, range_b, config_path="configs/dataset_config.yaml"):
    """Query LLM for puzzle 1 reasoning."""
    config = load_config(config_path)
    puzzle_config = config["puzzles"]["puzzle1"]
    
    # Use complete glob pattern from config
    images_path = puzzle_config["images_path"]
    reasoning_base_path = puzzle_config["reasoning_path"].replace("/*.txt", "")
    
    images = sorted(glob.glob(images_path))
    
    for image in tqdm(images[range_a:range_b], desc="Querying Puzzle 1"): 
        try:
            image_filename = image.split("/")[-1].split(".")[0]
            parts = image_filename.split("_")
            operation = "_".join(parts[-2:-1])
            value = parts[-1]
            
            reasoning_save_path = f"{reasoning_base_path}/{image_filename}.txt"
            
            if os.path.exists(reasoning_save_path):
                continue
            
            os.makedirs(os.path.dirname(reasoning_save_path), exist_ok=True)

            print("sending request")

            print(operation, value)
            
            output = send_request_with_retry(
                send_request_puzzle1, 
                image, operation, value, config, 
                config=config
            )
            
            with open(reasoning_save_path, "w", encoding="utf-8") as f:
                f.write(output)
                
        except Exception as e:
            print(f"Error processing {image}: {e}")


def query_puzzle2(range_a, range_b, config_path="configs/dataset_config.yaml"):
    """Query LLM for puzzle 2 reasoning."""
    config = load_config(config_path)
    puzzle_config = config["puzzles"]["puzzle2"]
    
    # Use complete glob pattern from config
    images_path = puzzle_config["images_path"]
    reasoning_base_path = puzzle_config["reasoning_path"].replace("/*.txt", "")
    
    images = sorted(glob.glob(images_path))
    
    for image in tqdm(images[range_a:range_b], desc="Querying Puzzle 2"):
        try:
            image_filename = image.split("/")[-1].split(".")[0]
            parts = image_filename.split("_")
            operation = parts[-6]
            order = "_".join(parts[-5:])
            
            reasoning_save_path = f"{reasoning_base_path}/{image_filename}.txt"
            
            if os.path.exists(reasoning_save_path):
                continue
                
            os.makedirs(os.path.dirname(reasoning_save_path), exist_ok=True)
            
            output = send_request_with_retry(
                send_request_puzzle2,
                image, operation, order, config,
                config=config
            )
            
            with open(reasoning_save_path, "w", encoding="utf-8") as f:
                f.write(output)
                
        except Exception as e:
            print(f"Error processing {image}: {e}")


def query_puzzle3(range_a, range_b, config_path="configs/dataset_config.yaml"):
    """Query LLM for puzzle 3 reasoning."""
    config = load_config(config_path)
    puzzle_config = config["puzzles"]["puzzle3"]
    generation_config = config["generation"]
    
    # Use complete glob pattern from config
    images_query_path = puzzle_config["images_query_path"]
    reasoning_base_path = puzzle_config["reasoning_path"].replace("/*.txt", "")
    configs_base_path = generation_config["output_dirs"]["puzzle3"]
    
    images = sorted(glob.glob(images_query_path))
    
    for image in tqdm(images[range_a:range_b], desc="Querying Puzzle 3"):
        try:
            image_filename = image.split("/")[-1].split(".")[0]
            operation_type = image.split("/")[-2]
            
            reasoning_save_path = f"{reasoning_base_path}/{image_filename}.txt"
            
            if os.path.exists(reasoning_save_path):
                continue
                
            os.makedirs(os.path.dirname(reasoning_save_path), exist_ok=True)
            
            # Use config path from generation settings
            configs_path = f"{configs_base_path}/{image_filename}.json"
            settings = json_to_text(configs_path)
            
            from .constants import OPERATION_MAP
            operation = OPERATION_MAP[operation_type]
            
            output = send_request_with_retry(
                send_request_puzzle3,
                image, settings, operation, operation_type, config,
                config=config
            )
            
            with open(reasoning_save_path, "w", encoding="utf-8") as f:
                f.write(output)
                
        except Exception as e:
            print(f"Error processing {image}: {e}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Query LLM for puzzle reasoning.")
    parser.add_argument("puzzle_type", type=str, choices=["1", "2", "3"], help="Puzzle type to query.")
    parser.add_argument("range_a", type=int, help="Start index of the range.")
    parser.add_argument("range_b", type=int, help="End index of the range.")
    parser.add_argument("--config", type=str, default="configs/dataset_config.yaml", help="Config file path.")
    args = parser.parse_args()
    
    if args.puzzle_type == "1":
        query_puzzle1(args.range_a, args.range_b, args.config)
    elif args.puzzle_type == "2":
        query_puzzle2(args.range_a, args.range_b, args.config)
    elif args.puzzle_type == "3":
        query_puzzle3(args.range_a, args.range_b, args.config)


if __name__ == "__main__":
    main()
