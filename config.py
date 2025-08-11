import json
import re
import math


white_balance_tone_contrast = [
    "Blacks",
    "Contrast",
    "Highlights",
    "Shadows",
    "Whites",
    "Exposure",
]
color_temperature = ["Temperature", "Tint", "Saturation"]
color_specific_adjustments = [
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
    "SaturationAdjustmentAqua",
    "SaturationAdjustmentBlue",
    "SaturationAdjustmentGreen",
    "SaturationAdjustmentYellow",
    "SaturationAdjustmentMagenta",
    "SaturationAdjustmentOrange",
    "SaturationAdjustmentPurple",
    "SaturationAdjustmentRed",
]
opertions = {
    0: white_balance_tone_contrast,
    1: color_temperature,
    2: color_specific_adjustments,
}


def adjust_value_accelerated(val):
    boost = (1 - math.log(1 + val) if val >= 0 else 0) / 20
    adjusted = math.log(1 + boost + val) if val >= 0 else 0
    return adjusted


def get_processed_predictions(stage, config, accrued_dehaze):
    re_config = {}
    ops = opertions[stage]
    for k, v in config.items():
        if k in ops:
            if k == "Temperature" and v < 0:
                # we assume operation invertability is equal during puzzle training but this is not always the case, temperature cooling is more harsh
                v = v // 2
            re_config[k] = v
            if k == "Saturation" and v < 0:
                accrued_dehaze += v
            if "Saturation" in k and v < 0:
                v = v // 3
            if k == "Saturation" and v > 0:
                accrued_dehaze += v // 3
                # v = int(v//3)
            # if "SaturationAdjustment" in k and v > 0:
            #     v = v//3
            if k == "Contrast" and v < 0:
                v = v // 3
            if k == "Contrast" and v > 0:
                accrued_dehaze += int(v * 1.5)
                # v = int(v//1.5)
            if k == "Exposure":
                v = v // 2
            if k == "Blacks":
                if v < 0:
                    accrued_dehaze += abs(v * 2)
                v = v // 2
            re_config[k] = v
        if stage == 1:
            accrued_dehaze = accrued_dehaze / 100
            accrued_dehaze = adjust_value_accelerated(accrued_dehaze)
            accrued_dehaze = min(0.6, accrued_dehaze)
            accrued_dehaze = accrued_dehaze * 100
            # re_config["Dehaze"] = accrued_dehaze
        if stage == 2 and accrued_dehaze > 0:
            saturation = re_config.get("Saturation", 0)
            saturation -= int(accrued_dehaze // 5)
            re_config["Saturation"] = saturation
    return re_config, accrued_dehaze


def process_values(config):
    processed_config = {"Contrast": 0, "Exposure": 0, "Saturation": 0}
    for key, value in config.items():
        key = re.sub(r"\d", "", key)
        value = value / 100.0
        processed_config[key] = value
    if processed_config.get("Blacks", 0) > 0:
        # we assume operation invertability is equal during puzzle training but this is not always the case, increasing blacks also needs some other subtle changes to mimimc opposite when blacks are decreased (other popular softwares use accompanying changes as well)
        processed_config["Blacks"] = processed_config["Blacks"] / 3
        processed_config["Contrast"] += processed_config["Blacks"] * 2
        processed_config["Exposure"] += processed_config["Blacks"] / 10
        processed_config["Saturation"] += processed_config["Blacks"] / 2
    print(processed_config)
    return processed_config


def modify_config_lines(config_path):
    """
    Read the config file and modify its values based on the rules.

    :param config_path: Path to the configuration file.
    :return: List of modified lines in the format "key=value".
    """
    config = read_config(config_path)
    config = json_to_text(config)
    config = config.split("\n")
    return config


def read_text(config_path):
    """
    Convert a list of modified config lines into a dictionary.

    :param modified_lines: List of config lines in the format "key=value".
    :return: Dictionary containing config keys and their corresponding values.
    """
    modified_lines = modify_config_lines(config_path)
    config = {}
    for line in modified_lines:
        key, value = line.split("=")
        config[key] = float(value)
    return process_values(config)


def read_json(config_path):
    with open(config_path) as f:
        config = json.load(f)
    config = json.dumps(config, ensure_ascii=False)
    config = json.loads(config)
    return process_values(config)


def read_config(config_path):
    if config_path.split(".")[-1] == "json":
        return read_json(config_path)
    return read_text(config_path)


def json_to_text(file_path):
    try:
        # Read the JSON file
        if type(file_path) == str:
            with open(file_path, "r") as file:
                data = json.load(file)
        else:
            data = file_path

        # Convert JSON to text paragraph
        paragraph = ""
        for key, value in data.items():
            if value == 0:
                continue
            if value > 0:
                value = "+" + str(value)
            paragraph += key + ": " + str(value) + "\n"

        return paragraph
    except FileNotFoundError:
        return "The specified file was not found."
    except json.JSONDecodeError:
        return "Error decoding JSON. Ensure the file contains valid JSON."
