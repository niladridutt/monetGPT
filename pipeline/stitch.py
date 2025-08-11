from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec


def merge_images_with_captions(
    image_paths, captions, output_path, text_file=None, font_size=10, dpi=300
):
    """
    Merge images side by side with captions below each image and optionally add a text column.

    :param image_paths: List of paths to the images.
    :param captions: List of captions corresponding to the images.
    :param output_path: Path to save the merged image.
    :param text_file: Optional path to a text file whose content is displayed as the 4th column.
    :param font_size: Font size for captions and text.
    """
    # Load images
    images = [mpimg.imread(path) for path in image_paths]

    # Calculate the number of columns
    num_cols = len(images) + (1 if text_file else 0)

    # Set up the figure and grid
    fig = plt.figure(figsize=(num_cols * 4, 6))  # Adjust width for each column
    gs = gridspec.GridSpec(
        1, num_cols, width_ratios=[1] * len(images) + ([0.9] if text_file else [])
    )
    plt.subplots_adjust(wspace=0.05)  # Minimal spacing between images

    # Plot images with captions
    for i, (img, caption) in enumerate(zip(images, captions)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(caption, fontsize=font_size, pad=10)

    # Add text column if a text file is provided
    if text_file:
        # Simple text file reading (removed config dependency)
        try:
            with open(text_file, 'r') as f:
                text_lines = f.readlines()
        except:
            text_lines = ["Text file not found"]
        
        text_column = "".join(text_lines)

        # Add a subplot for the text column
        ax = fig.add_subplot(gs[0, -1])
        ax.text(
            0.5,
            0.5,
            text_column,
            fontsize=font_size,
            ha="center",
            va="center",
            wrap=True,
            linespacing=1.7,
        )
        ax.axis("off")

    # Save the output
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Merged image with captions and optional text column saved at {output_path}")


def merge_images_side_by_side(image_path1, image_path2, output_path):
    """
    Merge two images side by side and save the output.

    :param image_path1: Path to the first image.
    :param image_path2: Path to the second image.
    :param output_path: Path to save the merged image.
    """
    # Open the two images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Find the height of the taller image
    max_height = max(img1.height, img2.height)

    # Resize images to the same height (maintaining aspect ratio)
    img1 = img1.resize((int(img1.width * max_height / img1.height), max_height))
    img2 = img2.resize((int(img2.width * max_height / img2.height), max_height))

    # Create a new image with a combined width
    combined_width = img1.width + img2.width
    merged_image = Image.new("RGB", (combined_width, max_height))

    # Paste the images into the new image
    merged_image.paste(img1, (0, 0))
    merged_image.paste(img2, (img1.width, 0))

    # Save the output
    merged_image.save(output_path)
    print(f"Merged image saved at {output_path}")


if __name__ == "__main__":
    # Example usage
    index = "1338_1"
    # merge_images_side_by_side(
    #     f"./ppr10k/src/{index}.tif",
    #     f"./ppr10k/target_a/{index}.tif",
    #     f"./ppr10k/merged/{index}.png",
    # )
    merge_images_side_by_side(
        f"./bird.tif",
        f"./test.tif",
        f"./merged.png",
    )
    
