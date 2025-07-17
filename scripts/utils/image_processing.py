"""
Image processing utilities for wavelet loss scripts.
"""

import hashlib
import numpy as np

# Optional image loading libraries
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None


def load_image(image_path, grayscale=False):
    """
    Load an image using either OpenCV or PIL, with optional grayscale conversion.

    Args:
        image_path (str): Path to the image file
        grayscale (bool): Whether to convert image to grayscale

    Returns:
        numpy.ndarray: Loaded image as a numpy array
    """
    # Prioritize OpenCV if available
    if cv2 is not None:
        # Read in BGR, optionally convert to grayscale
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Fall back to PIL if OpenCV is not available
    elif Image is not None:
        img = Image.open(image_path)
        if grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = np.array(img)

    else:
        raise ImportError("Neither OpenCV nor PIL is available for image loading")

    return img


def generate_image_hash(image_path, hash_length=8):
    """
    Generate a hash for the input image file.

    Args:
        image_path (str): Path to the image file
        hash_length (int): Length of the hash to return (default: 8)

    Returns:
        str: Hash string of the image
    """
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Generate SHA-256 hash and return first hash_length characters
    hash_obj = hashlib.sha256(image_data)
    return hash_obj.hexdigest()[:hash_length]
