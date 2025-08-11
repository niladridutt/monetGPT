from gimpfu import *
from image_ops.gimp_ops import *
import warnings
import json

warnings.filterwarnings("ignore")

def read_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

core_pipeline = {
    "Saturation": adjust_saturation,
    "Contrast": adjust_contrast,
    "Temperature": adjust_temperature,
    "Blacks": adjust_blacks,
    "Tint": adjust_tint,
}
light_operations = ["Shadows", "Highlights", "Whites"]

# these paths will get filled in by the calling script

config_path = 'data/puzzles1/configs/image 33_Tint_-70.json'
src_image_path = 'data/ppr10k/source/image 33.png'
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
            adjust_light(drawable,0,0,whites)
        
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
