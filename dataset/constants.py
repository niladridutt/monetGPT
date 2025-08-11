"""
Common constants and configurations for dataset generation.
"""

# Standard Lightroom operations
STANDARD_OPERATIONS = [
    "Blacks", "Exposure", "Highlights", "Shadows", "Whites", "Contrast",
    "Saturation", "Temperature", "Tint"
]

# Operation groupings for puzzle 3
WHITE_BALANCE_TONE_CONTRAST = ["Blacks", "Contrast", "Highlights", "Shadows", "Whites", "Exposure"]
COLOR_TEMPERATURE = ["Temperature", "Tint", "Saturation"]

# HSV color boundaries for image analysis
HSV_PREDEFINED_COLORS = {
    "red": [
        ((0, 100, 100), (10, 255, 255)),
        ((170, 100, 100), (180, 255, 255))
    ],
    "yellow": [((15, 100, 100), (35, 255, 255))],
    "orange": [((10, 100, 100), (20, 255, 255))],
    "green": [((35, 100, 100), (85, 255, 255))],
    "blue": [((95, 100, 100), (130, 255, 255))],
    "aqua": [((85, 100, 100), (95, 255, 255))],
    "purple": [((130, 100, 100), (150, 255, 255))],
    "magenta": [((150, 100, 100), (170, 255, 255))]
}

# Color-specific adjustments
COLOR_SPECIFIC_ADJUSTMENTS = [
    "HueAdjustmentAqua", "HueAdjustmentBlue", "HueAdjustmentGreen", "HueAdjustmentYellow", 
    "HueAdjustmentRed", "HueAdjustmentMagenta", "HueAdjustmentOrange", "HueAdjustmentPurple",
    "LuminanceAdjustmentAqua", "LuminanceAdjustmentBlue", "LuminanceAdjustmentGreen", 
    "LuminanceAdjustmentYellow", "LuminanceAdjustmentRed", "LuminanceAdjustmentOrange", 
    "LuminanceAdjustmentMagenta", "LuminanceAdjustmentPurple",
    "SaturationAdjustmentAqua", "SaturationAdjustmentBlue", "SaturationAdjustmentGreen", 
    "SaturationAdjustmentYellow", "SaturationAdjustmentMagenta", "SaturationAdjustmentOrange", 
    "SaturationAdjustmentPurple", "SaturationAdjustmentRed"
]

# Color operations mapping
COLOR_OPERATIONS_MAPPING = {
    "red": [op for op in COLOR_SPECIFIC_ADJUSTMENTS if "Red" in op],
    "yellow": [op for op in COLOR_SPECIFIC_ADJUSTMENTS if "Yellow" in op],
    "orange": [op for op in COLOR_SPECIFIC_ADJUSTMENTS if "Orange" in op],
    "green": [op for op in COLOR_SPECIFIC_ADJUSTMENTS if "Green" in op],
    "blue": [op for op in COLOR_SPECIFIC_ADJUSTMENTS if "Blue" in op],
    "aqua": [op for op in COLOR_SPECIFIC_ADJUSTMENTS if "Aqua" in op],
    "purple": [op for op in COLOR_SPECIFIC_ADJUSTMENTS if "Purple" in op],
    "magenta": [op for op in COLOR_SPECIFIC_ADJUSTMENTS if "Magenta" in op]
}

# Available operations mapping for puzzle 3
AVAILABLE_OPERATIONS = {
    "white-balance-tone-contrast": WHITE_BALANCE_TONE_CONTRAST,
    "color-temperature": COLOR_TEMPERATURE,
    "hsl": None
}

# Operation descriptions for puzzle 3
OPERATION_MAP = {
    "white-balance-tone-contrast": "**Exposure, Contrast & Tonal Range Adjustments** \n These parameters primarily affect the image's overall exposure, contrast, and distribution of tones (shadows, whites, blacks, and highlights). \n Operation selection available: [Blacks, Contrast, Highlights, Shadows, Whites, Exposure]",
    "color-temperature": "**White Balance & Global Saturation** \n These parameters adjust the photo's global color temperature (warmth/coolness), tint (green/magenta), and overall color intensity (saturation). \nOperation selection available: [Temperature, Tint, Saturation]", 
    "hsl": "**Selective Color Adjustments** \n These parameters allow fine-grained control over specific color ranges in the image, adjusting hue, saturation, or luminance for each color individually. \nOperation selection available: [HueAdjustmentAqua, HueAdjustmentBlue, HueAdjustmentGreen, HueAdjustmentYellow, HueAdjustmentRed, HueAdjustmentMagenta, HueAdjustmentOrange, HueAdjustmentPurple, LuminanceAdjustmentAqua, LuminanceAdjustmentBlue, LuminanceAdjustmentGreen, LuminanceAdjustmentYellow, LuminanceAdjustmentRed, LuminanceAdjustmentOrange, LuminanceAdjustmentMagenta, LuminanceAdjustmentPurple, SaturationAdjustmentAqua, SaturationAdjustmentBlue, SaturationAdjustmentGreen, SaturationAdjustmentYellow, SaturationAdjustmentMagenta, SaturationAdjustmentOrange, SaturationAdjustmentPurple, SaturationAdjustmentRed]. \nYou should only pick a few of these which are relevant for the image (You don't have to use all)"
}


# Summary mapping for puzzle 3
SUMMARY_MAP = {
    "white-balance-tone-contrast": """
Provide a concise explanation of the intended editing goal and the desired mood of the photo, considering how the original scene's lighting and time of day influence the necessary adjustments. Clearly compare the original image's characteristics with the edited version, explaining why specific changes are needed. Structure it as:

The original image was taken at [time of day: daytime, golden hour, night], featuring [lighting conditions: harsh sunlight, soft twilight glow, city streetlights], with [describe current state: overexposed highlights, underexposed shadows, low contrast, etc.]. Since it captures [describe scene: a mountain landscape, a bustling market, a concert stage, firworks], the editing should emphasize [key focus: bringing out details in shadows, correcting highlights, adding contrast] to match the natural atmosphere.
""",
    "color-temperature": """
Provide a concise explanation of the intended editing goal and the desired mood of the photo, clearly comparing the original image's characteristics with the edited version, and explaining why specific color-balance changes are needed. Structure it as:

The original image currently has [describe color imbalance: a strong blue cast, overly warm tones, under-saturated colors], which detracts from the intended look. Since it depicts [describe scene: a natural landscape, an indoor portrait session, a product display], the editing goal is to refine the overall color balance to achieve [desired mood: vibrant warmth, neutral realism, cool cinematic feel, etc.].
To accomplish this, adjustments like [list relevant operations] will help correct or enhance the photo's overall tone, ensuring the final image conveys [describe final impact: fresh and natural, soft and pleasing, bold and punchy, etc.].
""", 
    "hsl": """**Selective Color Adjustments**
Provide a concise explanation of the intended editing goal and the desired mood of the photo, clearly comparing the original image's color distribution with the edited version, explaining why specific selective color changes are needed. Structure it as:

The original image shows [describe specific color issues: oversaturated reds, dull greens, mismatched blues], which affects the overall harmony of the scene. Since it depicts [describe subject: a garden landscape, a fashion shoot, a neon-lit city street], the editing should target individual colors without altering the entire tone.
To achieve this, adjustments like [list relevant operations: adjusting hue for more natural skin tones, increasing saturation for a vibrant sky, lowering luminance of intense reds] will help create the intended effect of [describe final mood/look: balanced and cohesive, stylized and artistic, subtle and understated, etc.].
"""
}
