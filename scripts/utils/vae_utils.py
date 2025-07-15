"""
VAE utilities for wavelet loss scripts.
"""

import torch
import numpy as np

# Diffusers for VAE models
try:
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image
except ImportError:
    AutoencoderKL = None
    VaeImageProcessor = None
    Image = None


def load_vae_model(model_name_or_path="stabilityai/sd-vae-ft-mse"):
    """
    Load a VAE model from Hugging Face diffusers.
    
    Args:
        model_name_or_path (str): Model name or path to load
    
    Returns:
        tuple: (AutoencoderKL model, VaeImageProcessor)
    """
    if AutoencoderKL is None or VaeImageProcessor is None:
        raise ImportError("diffusers library is not installed. Install with: pip install diffusers")
    
    vae = AutoencoderKL.from_pretrained(model_name_or_path)
    
    # Get VAE scale factor from config (usually 8 for most VAE models)
    vae_scale_factor = getattr(vae.config, 'vae_scale_factor', 2 ** (len(vae.config.block_out_channels) - 1))
    
    # Create VaeImageProcessor
    processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor,
        do_resize=True,
        do_normalize=True,
        resample="bicubic"
    )
    
    return vae, processor


def preprocess_image_with_vae_processor(img, processor):
    """
    Preprocess image using VaeImageProcessor.

    Args:
        img (numpy.ndarray): Input image
        processor (VaeImageProcessor): Diffusers VAE image processor

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Convert numpy array to PIL Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Use VaeImageProcessor to preprocess the image
    processed = processor.preprocess(img)
    
    return processed


def encode_image_to_latent(vae, processor, img_tensor, device):
    """
    Encode image to VAE latent space.
    
    Args:
        vae (AutoencoderKL): VAE model
        processor (VaeImageProcessor): VAE image processor
        img_tensor (torch.Tensor): Preprocessed image tensor
        device (str): Device to use
    
    Returns:
        tuple: (latent, reconstructed_img, display_img)
    """
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        # Use diffusers VAE encode method
        latent_dist = vae.encode(img_tensor).latent_dist
        latent = latent_dist.sample()
        
        # Apply VAE scaling factor for proper latent space scaling
        latent = latent * vae.config.scaling_factor
        
        # Decode latent back to image (reverse scaling)
        latent_unscaled = latent / vae.config.scaling_factor
        reconstructed = vae.decode(latent_unscaled).sample
        
        # Process outputs using VaeImageProcessor
        reconstructed = processor.postprocess(reconstructed, output_type="pt")
        img_tensor_display = processor.postprocess(img_tensor, output_type="pt")
    
    return latent, reconstructed, img_tensor_display