import gradio as gr
import os
import tempfile
from PIL import Image
import numpy as np
import time
from model_handler import ESRGANHandler
from utils import process_image, get_file_size

# Initialize the model handler
model_handler = ESRGANHandler()

def upscale_image(input_image, scale_factor):
    """
    Upscale the input image using the ESRGAN model
    
    Args:
        input_image: Input image (PIL Image or numpy array)
        scale_factor: Scale factor for upscaling (2x, 4x)
    
    Returns:
        Tuple of (original image, upscaled image, processing time, original size, new size)
    """
    if input_image is None:
        return None, None, "No image provided", "N/A", "N/A"
    
    try:
        # Start timer
        start_time = time.time()
        
        # Convert to numpy array if it's a PIL Image
        if isinstance(input_image, Image.Image):
            input_np = np.array(input_image)
        else:
            input_np = input_image
            
        # Get original image size info
        original_h, original_w = input_np.shape[:2]
        original_size = get_file_size(input_np)
        
        # Process the image with the model
        upscaled_image = process_image(model_handler, input_np, int(scale_factor[0]))
        
        # Calculate processing time
        end_time = time.time()
        processing_time = f"{end_time - start_time:.2f} seconds"
        
        # Get upscaled image size info
        upscaled_h, upscaled_w = upscaled_image.shape[:2]
        upscaled_size = get_file_size(upscaled_image)
        
        size_info = f"Original: {original_w}×{original_h} ({original_size})\nUpscaled: {upscaled_w}×{upscaled_h} ({upscaled_size})"
        
        return input_image, upscaled_image, processing_time, size_info
        
    except Exception as e:
        return input_image, None, f"Error: {str(e)}", "Error occurred"


# Custom CSS for better styling
custom_css = """
.container {
    max-width: 1200px;
    margin: auto;
    padding-top: 1.5rem;
}

.header {
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
}

.header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.header p {
    font-size: 1.2rem;
    opacity: 0.9;
}

.tab-content {
    min-height: 500px;
    padding: 1rem;
}

.footer {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.9rem;
    color: #555;
}

.primary-button {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    border: none;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.primary-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.info-box {
    background-color: #f8f9fa;
    border-left: 4px solid #4b6cb7;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0 5px 5px 0;
}

.comparison-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    margin: 2rem 0;
}

.image-card {
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    padding: 1rem;
    transition: transform 0.3s ease;
}

.image-card:hover {
    transform: translateY(-5px);
}

.image-title {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: #333;
}

.scale-option {
    display: inline-block;
    margin-right: 10px;
    padding: 8px 15px;
    border: 2px solid #ddd;
    border-radius: 30px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.scale-option.selected {
    background-color: #4b6cb7;
    color: white;
    border-color: #4b6cb7;
}

.settings-box {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* Make Scale Factor label black */
.settings-box h4 {
    color: black !important;
}
"""

def create_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="Image Upscaler", theme=gr.themes.Soft(), css=custom_css) as interface:
        with gr.Column(elem_classes="container"):
            # Header Section
            with gr.Column(elem_classes="header"):
                gr.Markdown("# AI Image Upscaler")
                gr.Markdown("Transform your low-resolution images into stunning high-definition visuals using advanced AI")
            
            # Main Content
            with gr.Tabs() as tabs:
                # Upscaler Tab
                with gr.TabItem("Upscale Image", elem_classes="tab-content"):
                    with gr.Row():
                        # Left Column - Input
                        with gr.Column(scale=1):
                            gr.Markdown("### 📁 Upload Your Image")
                            input_image = gr.Image(
                                label="", 
                                type="pil",
                                elem_classes="image-card"
                            )
                            
                            gr.Markdown("### ⚙️ Settings")
                            with gr.Column(elem_classes="settings-box"):
                                gr.Markdown("#### Scale Factor")
                                scale_factor = gr.Radio(
                                    choices=["2", "4"], 
                                    value="4",
                                    label="",
                                    elem_classes="scale-option"
                                )
                                upscale_btn = gr.Button("✨ Enhance Image", variant="primary", elem_classes="primary-button")
                            
                            with gr.Accordion("📊 Technical Details", open=False):
                                processing_info = gr.Textbox(label="Processing Time", elem_classes="info-box")
                                size_info = gr.Textbox(label="Image Information", elem_classes="info-box")
                        
                        # Right Column - Output
                        with gr.Column(scale=1):
                            gr.Markdown("### 🖼️ Enhanced Result")
                            output_image = gr.Image(
                                label="", 
                                show_download_button=True,
                                elem_classes="image-card"
                            )
                
                # How to Use Tab
                with gr.TabItem("How to Use", elem_classes="tab-content"):
                    gr.Markdown("""
                    # How to Use the Image Upscaler
                    
                    ## Simple Steps
                    1. **Upload** your image using the uploader in the "Upscale Image" tab
                    2. **Select** your desired upscaling factor:
                       - 2x - Doubles the resolution (good for moderate enhancement)
                       - 4x - Quadruples the resolution (best for significant enhancement)
                    3. **Click** the "Enhance Image" button and wait for processing to complete
                    4. **Download** your enhanced image using the download button below the result
                    
                    ## Tips for Best Results
                    - For best results, use images with clear subjects and minimal noise
                    - Very blurry images may not show dramatic improvement
                    - Larger images will take longer to process
                    """)
                    
                # About Tab
                with gr.TabItem("About", elem_classes="tab-content"):
                    gr.Markdown("""
                    # About This Image Upscaler
                    
                    This application uses advanced image processing techniques with PyTorch and OpenCV to create high-quality image upscaling. The model enhances details while preserving the natural look of the image.
                    
                    ## Technologies Used
                    - **PyTorch**: For deep learning models
                    - **OpenCV**: For image processing
                    - **Gradio**: For the user interface
                    
                    ## How It Works
                    The upscaler uses a combination of bicubic interpolation and neural network enhancement to:
                    1. Increase the image resolution
                    2. Enhance fine details
                    3. Reduce artifacts and noise
                    4. Improve sharpness and clarity
                    """)
            
            # Footer
            with gr.Column(elem_classes="footer"):
                gr.Markdown("© 2025 AI Image Upscaler | Created with Gradio and PyTorch")
        
        # Set up the event handler
        upscale_btn.click(
            fn=upscale_image,
            inputs=[input_image, scale_factor],
            outputs=[input_image, output_image, processing_info, size_info],
            api_name="upscale"
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="localhost",
        server_port=5000,
        share=False,
        favicon_path=None
    )
