from PIL import Image
import numpy as np
import cv2

def apply_clahe_to_image(image, clip_limit=3, tile_grid_size=(10, 5)):
    """
    Apply CLAHE to the given image and return the result.
    """
    # Ensure the input is a numpy array
    if isinstance(image, Image.Image):
        # Convert PIL image to numpy array
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError("Input must be a PIL Image or a numpy array")

    # If the image is not single channel, convert it to grayscale
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ensure the image data type is np.uint8
    if img.dtype != np.uint8:
        img = np.uint8(img)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img)

    # Return the processed image as a PIL Image
    return Image.fromarray(img_clahe)