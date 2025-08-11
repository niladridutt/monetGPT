from PIL import Image
import numpy as np
from skimage import color
import cv2
import json
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2hsv, hsv2rgb
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import tifffile
from skimage.transform import resize
from . import image_dehazer
from config import read_config


def read_image(image_path):
    norm_factor = 255.0
    if image_path.endswith(".tif"):
        img = tifffile.imread(image_path)
        if img.dtype == np.uint8:
            norm_factor = 255.0
        elif img.dtype == np.uint16:
            norm_factor = 65535.0
        else:
            raise ValueError(f"Unsupported image bit depth: {img.dtype}")
        img = (img / norm_factor).astype(np.float32)
    else:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        img = img / 255.0
    return img, norm_factor


def save_tif(image, norm_factor, output_path):
    img = image * norm_factor
    if norm_factor == 255.0:
        dtype = np.uint8
        img = img.astype(np.uint8)
    else:
        dtype = np.uint16
        img = img.astype(np.uint16)
    # Save as uint16
    tifffile.imwrite(output_path, img, dtype=dtype)


def read_image_low_res(image_path, max_size=700):
    """
    Reads an image (TIFF or common 8-bit formats), normalizes it to [0, 1] in float32,
    and resizes it to have a maximum dimension of `max_size`.
    """
    norm_factor = 255.0
    if image_path.lower().endswith(".tif") or image_path.lower().endswith(".tiff"):
        # Read using tifffile to preserve 8-bit or 16-bit depth
        img = tifffile.imread(image_path)

        # Determine normalization factor dynamically
        if img.dtype == np.uint8:
            norm_factor = 255.0
        elif img.dtype == np.uint16:
            norm_factor = 65535.0
        else:
            raise ValueError(f"Unsupported TIFF bit depth: {img.dtype}")

        # Convert to float32 in [0, 1]
        img = (img / norm_factor).astype(np.float32)

        # Get current width and height from the NumPy array shape
        # (img can be HxW or HxWxC)
        height, width = img.shape[:2]

        # Compute the scale factor
        scale = min(max_size / width, max_size / height)

        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Resize using skimage to keep float32 data without clamping to 8-bit
            img_resized = resize(
                img,
                (new_height, new_width),
                preserve_range=True,  # keep data in [0,1]
                anti_aliasing=True,
            ).astype(np.float32)
        else:
            img_resized = img

        return img_resized, norm_factor

    else:
        # Non-TIFF branch (typical 8-bit images)
        pil_img = Image.open(image_path).convert("RGB")
        width, height = pil_img.size

        # Compute the scale factor
        scale = min(max_size / width, max_size / height)

        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Resize with PIL (this is already 8-bit, so no data-depth loss here)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to NumPy float32 in [0, 1]
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0

        return img_arr, norm_factor


def adjust_tint(image, intensity):
    """
    Adjust the tint of an image similar to Lightroom's tint slider.
    intensity: float in [-1.0, 1.0]
        -1.0 = More fluorescent green
         0.0 = No change
         1.0 = More magenta
    """
    if intensity == 0:
        return image
    # Clamp intensity
    intensity = np.clip(intensity, -1.0, 1.0)

    # Amplify intensity for a more pronounced effect
    intensity = intensity * 1.4

    # Load image and normalize pixel values

    # arr = image / 255.0  # Normalize to [0, 1]

    # Convert to Lab color space
    lab_arr = color.rgb2lab(image)

    # Adjust the 'a' channel for tint (green-magenta axis)
    a_shift = intensity * 10.0
    lab_arr[..., 1] += a_shift  # Adjust 'a' channel

    # Adjust the 'b' channel for balance (yellow-blue axis)
    # Stronger adjustment for green intensities to boost fluorescent effect
    if intensity < 0:
        b_shift = intensity * -10  # Brighten green tones
        l_shift = intensity * -10  # Slightly increase lightness for green
    else:
        b_shift = intensity * -30  # Subtle balance for magenta
        l_shift = intensity * 5.0  # Minor lightness tweak for magenta

    lab_arr[..., 2] += b_shift  # Adjust 'b' channel
    lab_arr[..., 0] += l_shift  # Adjust 'L' channel for brightness

    # Clip 'a', 'b', and 'L' channels to valid Lab ranges
    lab_arr[..., 0] = np.clip(lab_arr[..., 0], 0, 100)  # Lightness
    lab_arr[..., 1] = np.clip(lab_arr[..., 1], -128, 127)  # a channel
    lab_arr[..., 2] = np.clip(lab_arr[..., 2], -128, 127)  # b channel

    # Convert back to RGB
    result_arr = color.lab2rgb(lab_arr)

    # Clip values to [0, 1] and scale back to [0, 255]
    result_arr = np.clip(result_arr, 0.0, 1.0)
    return result_arr
    # result_arr = np.clip(result_arr, 0.0, 1.0) * 255.0

    # Convert back to PIL image
    # result_img = Image.fromarray(result_arr.astype("uint8"), mode="RGB")
    # return result_arr.astype("uint8")


def adjust_vibrance(image, t, sigma_s=0.2, sigma_v=0.2):
    """
    Apply a vibrance-like adjustment using smooth exponential (Gaussian) weighting
    centered around mid-tones for saturation and value, and introduce hue-based weighting
    to reduce changes to skin-tone hues, making the effect more similar to Lightroom's vibrance.

    Parameters:
        img_bgr (numpy.ndarray): Input image in BGR format (0-255).
        t (float): Vibrance intensity in the range [-1, 1].
                   Positive values increase vibrance, negative values decrease it.
        sigma_s (float): Standard deviation for Gaussian weighting on saturation.
        sigma_v (float): Standard deviation for Gaussian weighting on value.

    Returns:
        numpy.ndarray: Output image in RGB format with vibrance adjusted.
    """
    if t == 0:
        return image
    # Convert RGB to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Split channels
    H, S, V = cv2.split(img_hsv)

    # Normalize S and V to [0, 1]
    S_norm = S  # / 255.0
    V_norm = V  # / 255.0

    mean = 0.45

    # Compute Gaussian weights for saturation and value centered at 0.5
    w_s = np.exp(-((S_norm - mean) ** 2) / (2 * sigma_s**2))
    w_v = np.exp(-((V_norm - mean) ** 2) / (2 * sigma_v**2))

    # Combined saturation/value weight
    weight = w_s * w_v

    # Introduce hue-based weighting to protect skin tones.
    # Typical skin hues often cluster around H ~ 10 to 50 (on a 0-179 scale).
    # model a Gaussian "dip" around ~H=15 (0.08 normalized), reducing vibrance there.
    H_norm = H / 179.0
    mean_h_skin = 0.08  # Approx. hue ~15 degrees corresponds to common skin tones
    sigma_h = 0.05  # Adjust to control how broad the dip is around skin hues
    w_h = 1 - np.exp(-((H_norm - mean_h_skin) ** 2) / (2 * sigma_h**2))
    # At H_norm=mean_h_skin, w_h~0 (minimal vibrance), away from it w_h~1.

    # Combine all weights
    weight_total = weight * w_h

    # Apply vibrance formula using weight_total
    if t >= 0:
        # Increase vibrance more in less saturated areas:
        # S' = S + t*(1 - S) * weight_total
        S_new = S_norm + t * (1 - S_norm) * weight_total
    else:
        # Decrease vibrance proportionally:
        # S' = S*(1 + t*weight_total)
        S_new = S_norm * (1 + t * weight_total)
        # If weight_total=1, full effect; if weight_total=0, no change.

    # Clip result to [0,1]
    S_new = np.clip(S_new, 0, 1)

    # Convert S_new back to [0, 255]
    # S_final = (S_new * 255.0).astype(np.uint8)
    S_final = S_new

    # Merge channels back
    # img_hsv_final = cv2.merge([H.astype(np.uint8), S_final, V.astype(np.uint8)])
    img_hsv_final = cv2.merge([H, S_final, V])

    # Convert back to RGB (if you prefer BGR output, change COLOR_HSV2RGB to COLOR_HSV2BGR)
    img_final = cv2.cvtColor(img_hsv_final, cv2.COLOR_HSV2RGB)
    return img_final


def map_linear_to_exponential(user_input, min_exp=-2.5, max_exp=2.5):
    """
    Map user linear input (e.g., -5 to +5) to an exponential scale for exposure adjustment.
    - user_input: Linear input from the user (-5 to +5)
    - min_exp, max_exp: Internal exponential scaling range
    """
    # Map input (-5 to +5) to a normalized range (0 to 1)
    normalized = (user_input + 5) / 10  # Scale from -5..+5 to 0..1
    # Map normalized range to exponential range
    return min_exp + (max_exp - min_exp) * normalized


def adjust_exposure(image, intensity):
    """
    Simulate Lightroom-like exposure adjustment with user-friendly linear input (-5 to +5).
    """
    if intensity == 0:
        return image
    intensity = intensity / 1.3 * 5.0
    # Map user linear input to exponential exposure factor
    exposure_factor = map_linear_to_exponential(intensity)

    # Convert image to numpy array
    # image_array = np.array(image).astype(np.float32) / 255.0

    # Apply exposure adjustment with a tone-mapping curve
    adjusted = image * (2**exposure_factor)
    adjusted = np.clip(adjusted, 0, 1)  # Avoid clipping

    # Tone map the midtones for Lightroom-like effect
    midtone_curve = np.power(adjusted, 1 / (1.2 + exposure_factor * 0.2))
    adjusted = midtone_curve

    # Convert back to 8-bit
    # adjusted = (adjusted * 255).astype(np.uint8)
    return adjusted


##############################################################################
# 1. Vectorized RGB <-> HLS
##############################################################################
def rgb_to_hls_np(rgb):
    """
    Vectorized conversion from RGB to HLS, same definitions as colorsys.rgb_to_hls:
      - rgb: float32 or float64 array of shape (..., 3), in [0..1]
      - returns: H, L, S in [0..1], shape (..., 3)
    """
    # Extract R, G, B
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    L = (maxc + minc) / 2.0

    # Avoid division-by-zero:
    # delta = 0 => R=G=B => H=0 (arbitrary), S=0
    delta = maxc - minc
    small_delta_mask = delta < 1e-20

    # Hue
    # We'll fill H with zeros then update only where delta != 0
    H = np.zeros_like(L)

    # For the pixels where max is r:
    mask_r = (r == maxc) & (~small_delta_mask)
    H[mask_r] = (g[mask_r] - b[mask_r]) / delta[mask_r]
    # For the pixels where max is g:
    mask_g = (g == maxc) & (~small_delta_mask)
    H[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / delta[mask_g]
    # For the pixels where max is b:
    mask_b = (b == maxc) & (~small_delta_mask)
    H[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / delta[mask_b]

    # Scale H from [0..6] range into [0..1] range
    H = (H / 6.0) % 1.0  # ensure positive mod 1

    # Saturation
    # S = delta / (2L) if L<=0.5, else delta / (2 - 2L)
    # but we must handle delta=0 => S=0
    S = np.zeros_like(L)
    not_small_mask = ~small_delta_mask
    L_le_half = (L <= 0.5) & not_small_mask
    L_gt_half = (L > 0.5) & not_small_mask

    S[L_le_half] = (delta[L_le_half]) / (maxc[L_le_half] + minc[L_le_half])
    S[L_gt_half] = (delta[L_gt_half]) / (2.0 - (maxc[L_gt_half] + minc[L_gt_half]))

    # Combine
    hls = np.stack([H, L, S], axis=-1)
    return hls


def hls_to_rgb_np(hls):
    """
    Vectorized conversion from HLS to RGB, matching colorsys.hls_to_rgb:
      - hls: float32 or float64 array of shape (..., 3), each channel in [0..1]
      - returns: float array in shape (..., 3), each channel in [0..1]
    """
    H = hls[..., 0]
    L = hls[..., 1]
    S = hls[..., 2]

    # If S=0 => Gray => R=G=B=L
    rgb = np.zeros_like(hls)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]

    # For convenience, define a helper that adds a fractional hue:
    def hue_to_rgb(m1, m2, h):
        h_mod = h % 1.0
        # 6 segments in [0..1]
        return np.where(
            h_mod < 1 / 6,
            m1 + (m2 - m1) * 6 * h_mod,
            np.where(
                h_mod < 1 / 2,
                m2,
                np.where(h_mod < 2 / 3, m1 + (m2 - m1) * 6 * (2 / 3 - h_mod), m1),
            ),
        )

    # We only do the fancy stuff where S>0; otherwise it's just L.
    # We'll mask the S>0 part.
    s_pos_mask = S > 1e-20
    # Intermediate values
    m2 = np.where(L < 0.5, L + L * S, L + S - L * S)
    m1 = 2 * L - m2

    # Fill R,G,B only where S>0
    R[s_pos_mask] = hue_to_rgb(m1[s_pos_mask], m2[s_pos_mask], H[s_pos_mask] + 1 / 3)
    G[s_pos_mask] = hue_to_rgb(m1[s_pos_mask], m2[s_pos_mask], H[s_pos_mask])
    B[s_pos_mask] = hue_to_rgb(m1[s_pos_mask], m2[s_pos_mask], H[s_pos_mask] - 1 / 3)

    # Where S=0, it's simply gray => L
    gray_mask = ~s_pos_mask
    R[gray_mask] = L[gray_mask]
    G[gray_mask] = L[gray_mask]
    B[gray_mask] = L[gray_mask]

    return rgb


def adjust_hsl(
    image_array,
    color="red",
    hue_change=0.0,  # fraction of the max *range* hue shift ([-1..+1])
    saturation_change=0.0,  # fraction of the max *range* saturation shift ([-1..+1])
    luminance_change=0.0,  # fraction of the max *range* luminance shift ([-1..+1])
    blur_sigma=5.0,
):
    """
    Adjust hue, saturation, and luminance for a chosen color target, blending
    the effect spatially via a blurred mask and applying large shifts in
    multiple smaller passes to avoid abrupt changes.

    The dictionary below has two sets of limits for each channel:
        1) 'max_hue_shift_deg'        -> maximum single-pass hue shift (in degrees)
        2) 'max_range_h'             -> maximum total hue shift (in degrees)

        3) 'max_saturation_shift'     -> maximum single-pass saturation shift ([0..1])
        4) 'max_range_s'             -> maximum total saturation shift ([0..1])

        5) 'max_luminance_shift'      -> maximum single-pass luminance shift ([0..1])
        6) 'max_range_l'             -> maximum total luminance shift ([0..1])

    The user-supplied hue_change, saturation_change, luminance_change
    are fractions in [-1..+1] of the max_range_* for each channel.
    If the resulting shift is bigger than 'max_range_*', we clamp it.
    Then we break it into as many passes as needed, each not exceeding
    'max_*_shift' for that channel.

    For each pass:
      - We compute a hue-based mask for the target hue
      - We blur the mask to feather it.
      - We apply a small portion of the shift only to those pixels.

    Args:
        image_array (np.ndarray): (H, W, 3) RGB image in [0..255].
        color (str): Color key: "red", "green", "aqua", "magenta",
                     "yellow", "purple", "blue", "orange".
        hue_change (float):        [-1..+1], fraction of max_range_h to shift hue.
        saturation_change (float): [-1..+1], fraction of max_range_s to shift saturation.
        luminance_change (float):  [-1..+1], fraction of max_range_l to shift luminance.
        blur_sigma (float): Gaussian blur sigma for the mask.

    Returns:
        np.ndarray: (H, W, 3) uint8, the modified RGB image.
    """

    if hue_change == 0 and saturation_change == 0 and luminance_change == 0:
        return image_array

    # -------------------------------------------------------------
    # 1. Internal dictionary with channel limits per color
    # -------------------------------------------------------------

    # Load YAML file into a Python dictionary
    with open("configs/hsl.yaml", "r") as file:
        color_settings = yaml.safe_load(file)

    if color not in color_settings:
        print(f"WARNING: '{color}' is not in color_settings; using 'red' instead.")
        color = "red"

    cfg = color_settings[color]
    target_hue_deg = cfg["target_hue"]
    hue_range_deg = cfg["hue_range"]

    # Single-pass maxima
    max_hue_single_deg = cfg["max_hue_shift_deg"]
    max_sat_single = cfg["max_saturation_shift"]
    max_lum_single = cfg["max_luminance_shift"]

    # Overall maxima
    max_range_h = cfg["max_range_h"]
    # if hue_change < 0:
    #     max_range_h = cfg.get('min_range_h', max_range_h)
    max_range_s = cfg["max_range_s"]
    # if saturation_change < 0:
    #     max_range_s = cfg.get('min_range_s', max_range_s)
    max_range_l = cfg["max_range_l"]
    # if luminance_change < 0:
    #     max_range_l = cfg.get('min_range_l', max_range_l)

    max_hue_single_deg = max_range_h / 3
    max_sat_single = max_range_s / 3
    max_lum_single = max_range_l / 3

    # -------------------------------------------------------------
    # 2. Convert the image to HLS (vectorized) once
    # -------------------------------------------------------------
    # image_array is in [0..255], convert to [0..1]
    scaled_rgb = image_array  # .astype(np.float32) / 255.0
    hls_array = rgb_to_hls_np(scaled_rgb)  # shape (H, W, 3)
    # hls_array = rgb2hsv(scaled_rgb)[:, :, [0, 2, 1]]  # shape (H, W, 3)

    # -------------------------------------------------------------
    # 3. Function to split total desired shifts into increments
    # -------------------------------------------------------------
    def split_into_passes(total_shift, single_pass_max):
        """Break total_shift into increments each within ±single_pass_max."""
        if abs(total_shift) < 1e-10:
            return []
        sign = 1 if total_shift > 0 else -1
        magnitude = abs(total_shift)
        increments = []
        while magnitude > single_pass_max:
            increments.append(sign * single_pass_max)
            magnitude -= single_pass_max
        if magnitude > 1e-10:
            increments.append(sign * magnitude)
        return increments

    # -------------------------------------------------------------
    # 4. Compute total desired shifts, clamp, then break into passes
    # -------------------------------------------------------------
    # Hue
    desired_hue_total_deg = np.clip(hue_change * max_range_h, -max_range_h, max_range_h)
    hue_increments_deg = split_into_passes(desired_hue_total_deg, max_hue_single_deg)

    # Saturation
    desired_sat_total = np.clip(
        saturation_change * max_range_s, -max_range_s, max_range_s
    )
    sat_increments = split_into_passes(desired_sat_total, max_sat_single)

    # Luminance
    desired_lum_total = np.clip(
        luminance_change * max_range_l, -max_range_l, max_range_l
    )
    lum_increments = split_into_passes(desired_lum_total, max_lum_single)

    # -------------------------------------------------------------
    # 5. Helper function: build a blurred mask based on distance
    #    from target hue, then apply shift vectorized
    # -------------------------------------------------------------
    def apply_one_pass(hls_arr, hue_shift_deg=0.0, sat_shift=0.0, lum_shift=0.0):
        """
        1) Create a hue-distance mask from target_hue_deg
        2) Blur it (gaussian_filter)
        3) Apply the shift weighted by this mask
        """
        # shape (H, W, 3)
        H, W = hls_arr.shape[:2]

        # Hue in [0..1] => degrees in [0..360]
        hue_deg = (hls_arr[..., 0] * 360.0) % 360.0

        # Circular distance
        diff = np.abs(hue_deg - target_hue_deg)
        dist = np.minimum(diff, 360.0 - diff)

        # Cosine-based unblurred mask
        raw_mask = np.where(
            dist < hue_range_deg,
            np.maximum(0.0, np.cos(np.pi * dist / hue_range_deg)),
            0.0,
        )

        # Blur
        blurred_mask = gaussian_filter(raw_mask, sigma=blur_sigma) ** 1.3

        # Normalize mask so max is 1
        mx = blurred_mask.max()
        if mx > 1e-10:
            blurred_mask = blurred_mask / mx

        # Apply shifts
        if abs(hue_shift_deg) > 1e-10:
            frac_h = hue_shift_deg / 360.0
            hls_arr[..., 0] = (hls_arr[..., 0] + blurred_mask * frac_h) % 1.0

        if abs(sat_shift) > 1e-10:
            s_new = hls_arr[..., 2] + blurred_mask * sat_shift
            hls_arr[..., 2] = np.clip(s_new, 0.0, 1.0)

        if abs(lum_shift) > 1e-10:
            l_new = hls_arr[..., 1] + blurred_mask * lum_shift
            hls_arr[..., 1] = np.clip(l_new, 0.0, 1.0)

    # -------------------------------------------------------------
    # 6. Apply passes in order: hue, then saturation, then luminance
    # -------------------------------------------------------------
    for inc_deg in hue_increments_deg:
        apply_one_pass(hls_array, hue_shift_deg=inc_deg, sat_shift=0.0, lum_shift=0.0)

    for inc_s in sat_increments:
        apply_one_pass(hls_array, hue_shift_deg=0.0, sat_shift=inc_s, lum_shift=0.0)

    for inc_l in lum_increments:
        apply_one_pass(hls_array, hue_shift_deg=0.0, sat_shift=0.0, lum_shift=inc_l)

    # -------------------------------------------------------------
    # 7. Convert back to RGB, scale to [0..255], return uint8
    # -------------------------------------------------------------
    out_rgb_01 = hls_to_rgb_np(hls_array)  # shape (H, W, 3), in [0..1]
    out_rgb_01 = np.clip(out_rgb_01, 0, 1)
    # out_rgb_01 = hsv2rgb(hls_array[:, :, [0, 2, 1]])  # shape (H, W, 3), in [0..1]
    # out_rgb_255 = np.clip(out_rgb_01 * 255.0, 0, 255).astype(np.uint8)
    return out_rgb_01


def execute_hsl(config, image):
    colors = ["Red", "Yellow", "Green", "Aqua", "Blue", "Magenta", "Orange", "Purple"]
    adjustments_by_color = {}
    for color in colors:
        adjustments_by_color[color] = {
            "HueAdjustment": config.get("HueAdjustment" + color, 0),
            "SaturationAdjustment": config.get("SaturationAdjustment" + color, 0),
            "LuminanceAdjustment": config.get("LuminanceAdjustment" + color, 0),
        }
    for idx, color in enumerate(colors):
        color_settings = adjustments_by_color[color]
        image = adjust_hsl(
            image,
            color.lower(),
            color_settings["HueAdjustment"],
            color_settings["SaturationAdjustment"],
            color_settings["LuminanceAdjustment"],
        )
    return image


def smoothstep(x, edge0, edge1):
    """
    Smoothly interpolate from 0 to 1 as x goes from edge0 to edge1.
    For x < edge0, result = 0.
    For x > edge1, result = 1.
    Between edge0 and edge1, it's a cubic smooth step.
    """
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def create_mask(L, lower, upper, softness=0.1):
    """
    Returns a mask in [0..1] which is 0 below 'lower' and 0 above 'upper',
    but transitions smoothly in the boundary regions.
    The 'softness' is a fraction of the (upper-lower) range used for transitions.
    """
    mask = np.zeros_like(L, dtype=np.float32)

    # Ensure lower < upper
    if lower > upper:
        lower, upper = upper, lower

    # How big is the total range?
    span = upper - lower

    # We'll define two "transition" zones:
    #   [lower, lower + softness*span] for ramping from 0..1
    #   [upper - softness*span, upper] for ramping from 1..0
    transition = max(softness * span, 1e-8)

    # Region where we want the mask near 1.0
    # i.e., from (lower + transition) to (upper - transition)
    mid_low = lower + transition
    mid_high = upper - transition

    # 1) Ramp up from 0..1 as L goes from [lower..mid_low]
    ramp_up = smoothstep(L, lower, mid_low)

    # 2) Ramp down from 1..0 as L goes from [mid_high..upper]
    # We'll invert the smoothstep
    ramp_down = 1.0 - smoothstep(L, mid_high, upper)

    # Combine the two
    # In [lower..mid_low], ramp_up goes from 0..1; ramp_down is still 1
    # In [mid_low..mid_high], both ramp_up and ramp_down should be 1
    # In [mid_high..upper], ramp_down goes from 1..0; ramp_up is 1
    mask = np.minimum(ramp_up, ramp_down)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def adjust_tones(image, highlights=0.0, shadows=0.0, whites=0.0):
    """
    Adjusts the 'highlights', 'shadows', and 'whites' of an image (roughly)
    in a way inspired by Lightroom's tone controls, but with smoother,
    broader masks than a simple threshold.

    :param image:      Input image in BGR or RGB, shape (H, W, 3) as a NumPy array.
                       Should be in [0..255] if 8-bit, or [0..1] if float.
    :param highlights: Float in [-1..1]. Positive brightens highlights; negative darkens them.
    :param shadows:    Float in [-1..1]. Positive lifts shadows; negative crushes them.
    :param whites:     Float in [-1..1]. Positive raises extreme highlights; negative lowers them.
    :return:           Tone-adjusted image in the same range/type as input.
    """
    if highlights == 0 and shadows == 0:
        return image
    print(f"Executing highlights {highlights} shadows {shadows} whites {whites}")
    highlights = highlights * 100
    shadows = shadows * 100
    whites = whites * 100
    # 1) Convert to float in [0..1] if necessary
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    # 2) Split channels - assume RGB (adapt if your image is BGR)
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    # 3) Compute luminance (simple approximation)
    L = 0.299 * R + 0.587 * G + 0.114 * B

    # ----------------------------------------------------------------
    #        CREATE MASKS WITH WIDER COVERAGE & SMOOTHER BLENDING
    # ----------------------------------------------------------------
    # Example "wide" ranges (you can tweak these):
    #   Shadows:   affects ~ [0.0 .. 0.65]
    #   Highlights:affects ~ [0.35 .. 1.0]
    #   Whites:    narrower near the top, say [0.75 .. 1.0]
    # softness=0.2 means 20% of that range is used for transition
    # Adjust them to taste:

    # highlights = highlights/100 * 0.3 if highlights > 0 else highlights/100 * 0.2
    # shadows = shadows/100 * 0.15 if shadows > 0 else shadows/100 * 0.05
    # whites = whites/100 * 0.3
    # shadows_softness = 0.2 if shadows > 0 else 0.05
    # whites_softness = 2

    # INVERTIBLE
    highlights = highlights / 100 * 0.3
    shadows = shadows / 100 * 0.1
    whites = whites / 100 * 0.3
    shadows_softness = 0.13
    whites_softness = 2

    shadows_mask = create_mask(L, 0.0, 0.65, softness=shadows_softness)
    highlights_mask = create_mask(L, 0.35, 1.0, softness=0.5)
    whites_mask = create_mask(L, 0.65, 1.0, softness=whites_softness)

    # 4) Adjust Shadows
    #    We'll re-use the same concept as before: L' = L + factor * mask * some_scale
    #    The "some_scale" is how strong you want the slider to be.
    #    Typically a small value (0.5 or so) to avoid pushing too far.
    L_shadows = L + shadows * shadows_mask * 0.5

    # 5) Adjust Highlights
    L_highlights = L_shadows + highlights * highlights_mask * 0.5

    # 6) Adjust Whites
    L_whites = L_highlights + whites * whites_mask * 0.5

    # 7) Reconstruct channels
    #    Avoid dividing by zero
    eps = 1e-8
    ratio = L_whites / (L + eps)

    R_out = R * ratio
    G_out = G * ratio
    B_out = B * ratio

    out = np.stack([R_out, G_out, B_out], axis=-1)
    out = np.clip(out, 0.0, 1.0)

    # 8) Convert back to original dtype
    if image.dtype == np.uint8:
        out = (out * 255.0).astype(np.uint8)
    return out


def adjust_dehaze(image, norm_factor, delta):
    if delta <= 0.0:
        return image

    delta = min(0.63, delta)  # better to avoid too harsh effects
    dehazed_image, _ = image_dehazer.remove_haze(image, norm_factor, delta=delta)
    return dehazed_image


def execute_non_gimp_pipeline(config_path, image_path, output_path):
    config = read_config(config_path)
    image, norm_factor = read_image(image_path)
    core_pipeline = {
        "Tint": adjust_tint,
        "Vibrance": adjust_vibrance,
    }
    shadows = config.get("Shadows", 0)
    highlights = config.get("Highlights", 0)
    exposure = config.get("Exposure", 0)
    image = adjust_exposure(image, exposure)
    image = adjust_tones(image, shadows=shadows, highlights=highlights)
    dehaze_delta = config.get("Dehaze", 0)
    image = adjust_dehaze(image, norm_factor, dehaze_delta)
    for op, intensity in config.items():
        if op in core_pipeline.keys():
            operation = core_pipeline[op]
            image = operation(image, intensity)
            print("Executed ", op)
    image = execute_hsl(config, image)

    if output_path.endswith(".tif"):
        save_tif(image, norm_factor, output_path)
    else:
        # Convert to uint8 for PNG output
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
        output_image = Image.fromarray(image)
        output_image.save(output_path)


if __name__ == "__main__":
    config_path = ""
    image_path = ""
    output_path = ""
    execute_non_gimp_pipeline(config_path, image_path, output_path)
