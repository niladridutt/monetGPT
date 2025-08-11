"""
LLM-based image editing inference system for MonetGPT.
"""

import os
import base64
import json
import time
import io
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from openai import OpenAI
import yaml


class InferenceEngine:
    """Handles LLM-based image editing predictions and staged editing pipeline."""

    def __init__(self, inference_config_path: str = "configs/inference_config.yaml"):
        """Initialize the inference engine with configuration."""
        self.config = self._load_config(inference_config_path)
        self.client = self._initialize_client()

    def _load_config(self, config_path: str) -> dict:
        """Load inference configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _initialize_client(self) -> OpenAI:
        """Initialize the OpenAI client with configured settings."""
        api_key = os.environ.get(self.config["api"]["api_key_env"], "0")
        return OpenAI(api_key=api_key, base_url=self.config["api"]["base_url"])

    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 encoded PNG with optional resizing."""
        with Image.open(image_path) as img:
            # Resize if dimensions exceed configured maximum
            max_dim = self.config["image"]["max_dimension"]
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)

            # Convert to base64
            with io.BytesIO() as buffer:
                img.save(buffer, format=self.config["image"]["format"])
                image_data = buffer.getvalue()

        base64_encoded = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/{self.config['image']['format'].lower()};base64,{base64_encoded}"

    def _get_operation_description(self, operation: str) -> str:
        """Get formatted operation description for LLM prompt."""
        op_config = self.config["operations"][operation]
        available_ops = ", ".join(op_config["available"])

        description = f"{op_config['name']} \n {op_config['description']} \n"
        description += f"Operation selection available: [{available_ops}]"

        if "note" in op_config:
            description += f". \n{op_config['note']}"

        return description

    def _get_style_instruction(self, style: str = None) -> str:
        """Get style instruction based on configuration."""
        if style is None:
            style = self.config["default_style"]
        return self.config["styles"].get(style, "")

    def _create_analysis_prompt(
        self, operation: str, extra_instruction: str = "", style: str = None
    ) -> str:
        """Create the analysis prompt for the first stage."""
        operation_desc = self._get_operation_description(operation)
        short_operation = operation_desc.split("\n")[0]
        style_instruction = self._get_style_instruction(style)

        prompt = f"""
Analyze the provided image and develop a professional-grade editing plan using operations available in Adobe Lightroom to address issues in {operation_desc}. 

Your task is to identify all visual issues in the image and propose precise, optimized adjustments to address issues in {short_operation}. 

Create a professional editing plan for this photo with a list of the **optimal** adjustments needed to address the identified issues.

For each adjustment, follow this format:
Adjustment: [Mention the adjustment that needs to be made. Eg: **Adjustment:** The whites need to be greatly reduced**.]
Issue: [Explain the specific issue in the image, focusing on how it negatively impacts the photo, use as much context from the image as possible.]
Solution: [Describe how the adjustment will resolve the issue]

You should ensure that applying these adjustments will lead to an **optimal** image with balanced adjustment values specifically tuned for this image.

{style_instruction}

{extra_instruction}
"""
        return prompt.strip()

    def _create_json_prompt(
        self, extra_instruction: str = "", style: str = None
    ) -> str:
        """Create the JSON generation prompt for the second stage."""
        style_instruction = self._get_style_instruction(style)

        # Create intensity legend text
        legend_text = "The following legend can be used to map the intensity values from your previous answer:\n"
        for range_val, description in self.config["intensity_legend"].items():
            legend_text += f"{range_val}: {description}\n"

        prompt = f"""
Based on the editing plan and the original image, tell the **optimal** adjustment values needed to edit this photo in JSON format. All adjustment values are scaled between -100 and +100. You must ensure that the final edited image has **optimal** adjustment values to look like an **optimal image**.

{legend_text}

{style_instruction}

{extra_instruction}
"""
        return prompt.strip()

    def query_llm_for_adjustments(
        self,
        image_path: str,
        operation: str,
        extra_instruction: str = "",
        style: str = None,
        timeout: int = 0,
    ) -> Tuple[List[str], bool]:
        """Query the LLM for image editing adjustments."""

        # Encode image
        encoded_image = self.encode_image_to_base64(image_path)

        # Create prompts
        analysis_prompt = self._create_analysis_prompt(
            operation, extra_instruction, style
        )
        json_prompt = self._create_json_prompt(extra_instruction, style)

        # First round: Analysis
        messages = [
            {
                "role": "system",
                "content": "You are a helpful advanced image-editing assistant with expertise in Adobe Lightroom.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": analysis_prompt},
                    {"type": "image_url", "image_url": {"url": encoded_image}},
                ],
            },
        ]

        print("Sending analysis request...")
        result = self.client.chat.completions.create(
            messages=messages,
            model=self.config["api"]["model"],
            temperature=self.config["api"]["temperature"],
        )

        # Save temporary result for debugging
        temp_file = self.config["output"]["temp_result_file"]
        with open(temp_file, "w") as f:
            f.write(result.choices[0].message.content)

        # Add timeout if specified
        if timeout > 0:
            print(f"Waiting for {timeout} seconds...")
            time.sleep(timeout)

        # Read back the result
        with open(temp_file, "r") as f:
            analysis_output = f.read()

        print("Analysis result:", analysis_output)

        answers = [analysis_output]

        # Check if no adjustments are needed
        if "**no further adjustments are needed**" in analysis_output.lower():
            return answers, False

        # Second round: JSON generation
        messages.append({"role": "assistant", "content": analysis_output})
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": json_prompt}]}
        )

        result = self.client.chat.completions.create(
            messages=messages,
            model=self.config["api"]["model"],
            temperature=self.config["api"]["temperature"],
        )

        json_output = result.choices[0].message.content
        answers.append(json_output)
        print("JSON result:", json_output)

        return answers, True

    def extract_json_from_response(self, response: str) -> dict:
        """Extract and parse JSON from LLM response."""
        try:
            start_index = response.find("{")
            end_index = response.rfind("}")
            json_str = response[start_index : end_index + 1]
            json_str = json_str.replace("\\", "")
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Error parsing JSON from response: {e}")
            return {}

    def adjustments_to_text(self, adjustments: dict) -> str:
        """Convert adjustments dictionary to human-readable text."""
        if not adjustments:
            return "No adjustments"

        text = ""
        for key, value in adjustments.items():
            if value == 0:
                continue
            if value > 0:
                value = f"+{value}"
            text += f"{key}: {value}\n"

        return text.strip()


class StagedEditingPipeline:
    """Manages the staged editing pipeline with LLM predictions and edits."""

    def __init__(
        self,
        inference_config_path: str = "configs/inference_config.yaml",
        pipeline_config_path: str = "configs/pipeline_config.yaml",
    ):
        """Initialize the staged editing pipeline."""
        self.inference_engine = InferenceEngine(inference_config_path)
        self.config = self.inference_engine.config

        # Import pipeline utilities at module level to avoid import issues
        import sys
        import os

        sys.path.append(os.path.dirname(os.path.abspath(__file__ + "/..")))

        from pipeline.utils import load_combined_config
        from pipeline.core import ImageEditingPipeline


        # from run_pipeline_from_config import execute_edit
        from config import get_processed_predictions

        self.pipeline_config = load_combined_config(pipeline_config_path)
        self.image_execution_engine = ImageEditingPipeline()
        self.execute_edit = self.image_execution_engine.execute_edit
        self.get_processed_predictions = get_processed_predictions

    def process_image(
        self,
        image_path: str,
        output_base_path: str,
        style: str = None,
        extra_instructions: Dict[str, str] = None,
    ) -> Tuple[str, dict, str]:
        """
        Process an image through the complete staged editing pipeline.

        Args:
            image_path: Path to input image
            output_base_path: Base path for outputs (without extension)
            style: Editing style ('balanced', 'vibrant', 'retro')
            extra_instructions: Dictionary of operation-specific extra instructions

        Returns:
            Tuple of (final_image_path, final_adjustments, reasoning_text)
        """
        if extra_instructions is None:
            extra_instructions = {}

        # Get operations sequence from config
        operations = self.config["processing"]["operations_sequence"]
        save_after_each_stage = self.config["processing"]["save_after_each_stage"]

        # Initialize tracking variables
        current_image_path = image_path
        final_adjustments = {}
        all_reasoning = ""
        accrued_dehaze = 0

        # Process each operation stage
        for stage, operation in enumerate(operations):
            print(f"Executing stage {stage + 1}: {operation}")

            # Determine output path for this stage
            if save_after_each_stage:
                stage_output_path = f"{output_base_path}_{operation}{self.config['output']['image_extension']}"
            else:
                stage_output_path = (
                    f"{output_base_path}{self.config['output']['image_extension']}"
                )

            # Get extra instruction for this operation
            extra_instruction = extra_instructions.get(operation, "")

            # Query LLM for adjustments
            outputs, is_editing_needed = (
                self.inference_engine.query_llm_for_adjustments(
                    current_image_path, operation, extra_instruction, style
                )
            )

            # Accumulate reasoning
            stage_reasoning = "\n".join(outputs)
            all_reasoning += (
                f"\n\n=== Stage {stage + 1}: {operation} ===\n\n{stage_reasoning}"
            )

            # Process adjustments if editing is needed
            if is_editing_needed and len(outputs) > 1:
                json_response = outputs[-1]
                stage_adjustments = self.inference_engine.extract_json_from_response(
                    json_response
                )

                if stage_adjustments:
                    # Apply predictions using existing logic
                    processed_adjustments, accrued_dehaze = self.get_processed_predictions(
                        stage, stage_adjustments, accrued_dehaze
                    )

                    if stage == 2:
                        processed_adjustments["Dehaze"] = int(accrued_dehaze)
                    # Update final adjustments
                    final_adjustments.update(processed_adjustments)

                    # Save stage config
                    stage_config_path = f"{output_base_path}_{operation}{self.config['output']['config_extension']}"
                    with open(stage_config_path, "w") as f:
                        json.dump(processed_adjustments, f, indent=4)

                    # Execute edit
                    self.execute_edit(
                        stage_config_path, current_image_path, stage_output_path
                    )

                    print(f"Stage {stage + 1} adjustments:")
                    print(self.inference_engine.adjustments_to_text(stage_adjustments))

            # Update current image path for next stage
            if os.path.exists(stage_output_path):
                current_image_path = stage_output_path

        # Save final results
        final_config_path = (
            f"{output_base_path}{self.config['output']['config_extension']}"
        )
        final_reasoning_path = (
            f"{output_base_path}{self.config['output']['reasoning_extension']}"
        )

        with open(final_config_path, "w") as f:
            json.dump(final_adjustments, f, indent=4)

        with open(final_reasoning_path, "w") as f:
            f.write(all_reasoning)

        print("\nFinal processed adjustments:")
        print(self.inference_engine.adjustments_to_text(final_adjustments))

        return current_image_path, final_adjustments, all_reasoning


# Convenience functions for backward compatibility
def process_single_image(
    image_path: str,
    output_base_path: str,
    style: str = "balanced",
    extra_instructions: Dict[str, str] = None,
) -> Tuple[str, dict, str]:
    """Process a single image through the staged editing pipeline."""
    pipeline = StagedEditingPipeline()
    return pipeline.process_image(
        image_path, output_base_path, style, extra_instructions
    )


def batch_process_images(
    image_paths: List[str],
    output_dir: str,
    style: str = "balanced",
    extra_instructions: Dict[str, str] = None,
) -> List[Tuple[str, dict, str]]:
    """Process multiple images through the staged editing pipeline."""
    pipeline = StagedEditingPipeline()
    results = []

    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_base_path = os.path.join(output_dir, image_name)

        try:
            result = pipeline.process_image(
                image_path, output_base_path, style, extra_instructions
            )
            results.append(result)
            print(f"Successfully processed: {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append((None, {}, ""))

    return results
