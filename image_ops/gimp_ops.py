from gimpfu import *
import numpy as np


def adjust_tint(image, drawable, intensity):
    """
    Adjust the image tint based on intensity.
    intensity: float in [-1.0, 1.0]
        -1.0 = More green
         0.0 = No change
         1.0 = More magenta
    """
    # Ensure intensity is within [-1, 1]
    intensity = max(-1.0, min(1.0, intensity))
    # Scaling factor to translate intensity to color balance value
    factor = 30.0
    # Other axes remain unchanged
    cyan_red = 0.0
    magenta_green = intensity * factor  # +ve: more magenta, -ve: more green
    yellow_blue = 0.0
    transfer_mode = 1  # Affect mid-tones
    preserve_lum = 1
    # Apply the color balance to affect tint
    pdb.gimp_drawable_color_balance(
        drawable, transfer_mode, preserve_lum, cyan_red, magenta_green, yellow_blue
    )


def adjust_temperature(image, drawable, intensity):
    """
    Adjust the image temperature based on intensity.
    intensity: float in [-1.0, 1.0]
        -1.0 = Very cool (more cyan/blue)
         0.0 = No change
         1.0 = Very warm (more red/yellow)
    """
    intensity = intensity * 2.5
    # Ensure intensity is within [-1, 1]
    # intensity = max(-1.0, min(1.0, intensity))
    # Chosen scaling factor to translate intensity to color balance values
    red_factor = 15.0
    yellow_factor = 30.0
    # Positive intensity: warm (more red, more yellow)
    # Negative intensity: cool (more cyan, more blue)
    cyan_red = intensity * red_factor  # +ve: more red, -ve: more cyan
    magenta_green = 0.0  # No change
    yellow_blue = intensity * (
        -yellow_factor
    )  # +ve: more yellow (because negative sign), -ve: more blue
    transfer_mode = 1  # 0=Shadows, 1=Midtones, 2=Highlights
    preserve_lum = 1
    # Apply color balance
    pdb.gimp_drawable_color_balance(
        drawable, transfer_mode, preserve_lum, cyan_red, magenta_green, yellow_blue
    )
    transfer_mode = 2
    pdb.gimp_drawable_color_balance(
        drawable, transfer_mode, preserve_lum, cyan_red, magenta_green, yellow_blue
    )
    transfer_mode = 0
    pdb.gimp_drawable_color_balance(
        drawable, transfer_mode, preserve_lum, cyan_red, magenta_green, yellow_blue
    )



def adjust_clarity(image, drawable, intensity):
    """
    Adjust the clarity of the image.
    intensity: float in [-1.0, 1.0]
      -1.0 = Decrease clarity (soften)
       0.0 = No change
       1.0 = Increase clarity (add local contrast)
    """
    # Clamp intensity
    intensity = max(-1.0, min(1.0, intensity))
    if intensity == 0.0:
        # No change needed
        return
    if intensity > 0:
        # Positive clarity: Use Unsharp Mask with a large radius and moderate amount.
        # For intensity=1.0, use radius ~30 and amount ~50.
        # For less than full intensity, scale accordingly.
        radius = 30.0
        amount = intensity * 0.45
        threshold = 0
        pdb.plug_in_unsharp_mask(image, drawable, radius, amount, threshold)
        saturation = intensity * 20
        pdb.gimp_drawable_hue_saturation(drawable, 0, 0, 0, -saturation, 0)
    else:
        # Negative clarity: Mild Gaussian blur to reduce local contrast.
        # Scale blur radius with intensity. For intensity=-1.0, radius=5.
        blur_radius = abs(intensity) * 5.0
        pdb.plug_in_gauss_rle(
            image,
            drawable,
            blur_radius,
            True,  # Blur horizontally
            True,  # Blur vertically
        )


def adjust_light(
    drawable, shadows, highlights, whites, compress=75, shadow_correct=50, hl_correct=50
):
    radius = 300
    shadows = shadows * 100
    highlights = highlights * 100
    whitepoint_factor = 2.7 if whites > 0 else 1.5
    whites = int(whites * 10 * whitepoint_factor)
    whitepoint_sign = 1 if whites >= 0 else -1
    whites = abs(whites)
    white_0 = min(whites, 10) * whitepoint_sign
    pdb.gimp_drawable_shadows_highlights(
        drawable,  # Drawable
        shadows,  # Shadows adjustment
        highlights,  # Highlights adjustment
        white_0,  # Whitepoint adjustment
        radius,  # Radius (default value)
        compress,  # Compress (default value)
        shadow_correct,  # Shadows color correction (default value)
        hl_correct,  # Highlights color correction (default value)
    )
    whites = whites - 10
    shadows = 0
    highlights = 0
    while whites > 0:
        residue = min(whites, 10) * whitepoint_sign
        pdb.gimp_drawable_shadows_highlights(
            drawable,  # Drawable
            shadows,  # Shadows adjustment
            highlights,  # Highlights adjustment
            residue,  # Whitepoint adjustment
            radius,  # Radius (default value)
            compress,  # Compress (default value)
            shadow_correct,  # Shadows color correction (default value)
            hl_correct,  # Highlights color correction (default value)
        )
        whites = whites - 10


def adjust_contrast(image, drawable, intensity):
    # - range (-0.2,0.2) contrast
    print("intensssity", intensity)
    pdb.gimp_drawable_brightness_contrast(drawable, 0, intensity / 5)


def adjust_saturation(image, drawable, intensity):
    # range (-100,80) saturation
    pdb.gimp_drawable_hue_saturation(drawable, 0, 0, 0, int(intensity * 100), 0)


def adjust_sharpen(image, drawable, intensity):
    if intensity > 0:
        pdb.plug_in_sharpen(image, drawable, 70)
    else:
        # blur_radius range (0,4) -> 0,150
        blur_radius = int(abs(intensity) * 4)
        pdb.plug_in_gauss_rle(image, drawable, blur_radius, True, True)


def adjust_shadows_highlights(
    image,
    drawable,
    adjust_type,
    intensity,
    compress=75,
    shadow_correct=50,
    hl_correct=50,
):
    """
    Adjust shadows, highlights, or whites of the given drawable.

    Parameters:
        drawable: GIMP drawable object.
        adjust_type: str, one of "shadows", "highlights", or "whitepoint".
        intensity: float, value scaled between -1 and +1 for adjustment intensity.
    Returns:
        None
    """
    # Ensure intensity is scaled between -1 and +1
    if not (-1.0 <= intensity <= 1.0):
        raise ValueError("Intensity must be between -1 and +1.")
    # Map intensity to the appropriate ranges for each parameter
    if adjust_type == "shadows":
        shadows = intensity * 100  # Scale to -100 to +100
        highlights = 0
        whitepoint = 0
    elif adjust_type == "highlights":
        shadows = 0
        highlights = intensity * 100  # Scale to -100 to +100
        whitepoint = 0
    elif adjust_type == "whites":
        shadows = 0
        highlights = 0
        # whitepoint_factor = 2.5 if intensity > 0 else 1.5
        # INVERTIBLE
        whitepoint_factor = 2.0
        whitepoint = int(intensity * 10 * whitepoint_factor)  # Scale to -10 to +10
    else:
        raise ValueError(
            "Invalid adjust_type. Must be 'shadows', 'highlights', or 'whites'."
        )
    radius = 300
    if whitepoint >= 0:
        # Call the internal GIMP function
        while whitepoint >= 0:
            whitepoint_val = min(whitepoint, 10)
            pdb.gimp_drawable_shadows_highlights(
                drawable,  # Drawable
                shadows,  # Shadows adjustment
                highlights,  # Highlights adjustment
                min(whitepoint, 10),  # Whitepoint adjustment
                radius,  # Radius (default value)
                compress,  # Compress (default value)
                shadow_correct,  # Shadows color correction (default value)
                hl_correct,  # Highlights color correction (default value)
            )
            whitepoint = whitepoint - 10
            # if whitepoint > 0:
            #     adjust_tint_lab(drawable, whitepoint_val/4)
    else:
        while whitepoint < 0:
            whitepoint_val = max(whitepoint, -10)
            pdb.gimp_drawable_shadows_highlights(
                drawable,  # Drawable
                shadows,  # Shadows adjustment
                highlights,  # Highlights adjustment
                whitepoint_val,  # Whitepoint adjustment
                radius,  # Radius (default value)
                compress,  # Compress (default value)
                shadow_correct,  # Shadows color correction (default value)
                hl_correct,  # Highlights color correction (default value)
            )
            whitepoint = whitepoint + 10
            # if whitepoint < 0:
            #     adjust_tint_lab(drawable, whitepoint_val/4)


def adjust_blacks(image, drawable, intensity):
    """
    Adjust blacks using a spline curve.
    intensity: float between -1 and +1
        -1 -> deeper blacks
         0 -> no change (straight line)
        +1 -> lifted blacks
    """
    y = 30
    # INVERTIBLE
    if intensity < 0:
        x = y + ((abs(intensity) + 1) ** 4) * 6
    else:
        x = y + ((abs(intensity) + 1) ** 4) * 6
        x, y = y, x
    controls = [0, 0, x, y, 255, 255]
    # Normalize controls for gimp_drawable_curves_spline (0..1 range)
    normalized_controls = [c / 255.0 for c in controls]
    # channel=0 means HISTOGRAM-VALUE channel
    pdb.gimp_drawable_curves_spline(
        drawable, 0, len(normalized_controls), normalized_controls
    )


