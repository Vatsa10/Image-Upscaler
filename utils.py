import cv2
import numpy as np
from PIL import Image
import io

def process_image(model_handler, input_image, scale_factor=4):
    """
    Process an input image for upscaling
    
    Args:
        model_handler: The ESRGANHandler instance
        input_image: Input image as numpy array (RGB)
        scale_factor: Scale factor for upscaling
    
    Returns:
        Upscaled image as numpy array
    """
    # Convert RGB to BGR (OpenCV format)
    img_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Upscale the image
    result_bgr = model_handler.upscale(img_bgr, outscale=scale_factor)
    
    # Convert back to RGB
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    
    return result_rgb

def get_file_size(img_array):
    """
    Calculate the approximate file size of an image
    
    Args:
        img_array: Image as numpy array
    
    Returns:
        String representation of file size
    """
    # Convert to PIL image
    img = Image.fromarray(img_array.astype(np.uint8))
    
    # Save to BytesIO to get size
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    size_bytes = img_byte_arr.getbuffer().nbytes
    
    # Convert to appropriate unit
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def handle_file_upload(file_obj):
    """
    Handle file upload and convert to numpy array
    
    Args:
        file_obj: File object from upload
    
    Returns:
        Image as numpy array
    """
    try:
        # Read image
        img = Image.open(file_obj)
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error handling file upload: {e}")
        raise
