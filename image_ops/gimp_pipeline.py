from gimpfu import *
import sys
import os
# Add the image_ops directory to the path to import gimp_ops
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from gimp_ops import *
import warnings
import json

# Add parent directory to import config
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from config import read_config

warnings.filterwarnings("ignore")

core_pipeline = {
    "Saturation": adjust_saturation,
    "Contrast": adjust_contrast,
    "Temperature": adjust_temperature,
    "Blacks": adjust_blacks,
    "Tint": adjust_tint,
}
light_operations = ["Shadows", "Highlights", "Whites"]

# DO NOT MODIFY; these paths will get filled in by the calling script

config_path = 'data/puzzles1/configs/image 34_HueAdjustmentOrange_-77.json'
src_image_path = 'data/ppr10k/source/image 34.png'
output_image_path = './ppr10k/export/temp.png'


def execute(image_path, output_path):
    try:
        print("Loading image:", image_path)
        image = pdb.gimp_file_load(image_path, image_path)
        image = gimp.image_list()[0]
        drawable = pdb.gimp_image_get_active_drawable(image)
        print("Image loaded successfully")

        print("Reading config:", config_path)
        config = read_config(config_path)
        print("Config loaded:", config)

        for op, intensity in config.items():
            if op in core_pipeline.keys():
                operation = core_pipeline[op]
                operation(image, drawable, intensity)
                print("Executed ", op, intensity)
        whites = config.get("Whites", 0)
        if whites != 0:
            adjust_light(drawable, 0, 0, whites)

        print("Saving image to:", output_path)
        pdb.gimp_file_save(image, drawable, output_path, output_path)
        print("Image saved successfully")

    except Exception as e:
        print("Error in execute function:", str(e))
        import traceback

        traceback.print_exc()


def main():
    execute(src_image_path, output_image_path)


main()
