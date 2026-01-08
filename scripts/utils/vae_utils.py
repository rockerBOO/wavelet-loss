"""
VAE utilities for wavelet loss scripts.
"""

import torch
import numpy as np

# Diffusers for VAE models
try:
    from diffusers import AutoencoderKL, AutoencoderKLQwenImage, AutoencoderDC
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image
except ImportError:
    AutoencoderKL = None
    VaeImageProcessor = None
    Image = None

# Optional image processing libraries
try:
    import cv2
except ImportError:
    cv2 = None


def load_vae_model(model_name_or_path="stabilityai/sd-vae-ft-mse", subfolder=None):
    """
    Load a VAE model from Hugging Face diffusers.

    Args:
        model_name_or_path (str): Model name or path to load

    Returns:
        tuple: (AutoencoderKL model, VaeImageProcessor)
    """
    if AutoencoderKL is None or VaeImageProcessor is None:
        raise ImportError("diffusers library is not installed.")

    if "qwen" in model_name_or_path.lower():
        vae = AutoencoderKLQwenImage.from_pretrained(model_name_or_path, subfolder=subfolder)
        vae_scale_factor = 2 ** len(vae.temperal_downsample)
    else:
        vae = AutoencoderKL.from_pretrained(model_name_or_path, subfolder=subfolder)
        # Get VAE scale factor from config (usually 8 for most VAE models)
        vae_scale_factor = getattr(vae.config, "vae_scale_factor", 2 ** (len(vae.config.block_out_channels) - 1))

    # Create VaeImageProcessor
    processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor, do_resize=True, do_normalize=True, resample="bicubic"
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

        # if hasattr(vae.config, "scaling_factor"):
        #     # Apply VAE scaling factor for proper latent space scaling
        #     latent = latent * vae.config.scaling_factor
        #
        #     # Decode latent back to image (reverse scaling)
        #     latent_unscaled = latent / vae.config.scaling_factor

        reconstructed = vae.decode(latent).sample

        # Process outputs using VaeImageProcessor
        reconstructed = processor.postprocess(reconstructed, output_type="pt")
        img_tensor_display = processor.postprocess(img_tensor, output_type="pt")

    return latent, reconstructed, img_tensor_display


def next_power_of_two(x):
    """
    Find the next power of two for a given number.
    Args:
        x (int): Input number
    Returns:
        int: Next power of two
    """
    return 2 ** (x - 1).bit_length()


def preprocess_image(img, target_size=None, normalize=True, force_power_of_two=True, min_size=256):
    """
    Preprocess image for wavelet loss calculation.
    Args:
        img (numpy.ndarray): Input image
        target_size (tuple, optional): Resize image to this size
        normalize (bool): Normalize image to [0, 1] range
        force_power_of_two (bool): Ensure image dimensions are powers of two
        min_size (int): Minimum size for image dimensions
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Resize if target size is specified
    if target_size:
        if cv2 is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        elif Image is not None:
            img = np.array(Image.fromarray(img).resize(target_size, Image.LANCZOS))
    
    # Ensure power of two dimensions if requested
    if force_power_of_two and len(img.shape) >= 2:
        height, width = img.shape[-2:]
        # Ensure minimum size
        new_height = max(next_power_of_two(height), min_size)
        new_width = max(next_power_of_two(width), min_size)
        # Pad or resize to power of two
        if cv2 is not None:
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        elif Image is not None:
            img = np.array(Image.fromarray(img).resize((new_width, new_height), Image.LANCZOS))
    
    # Add batch and channel dimensions if needed
    if len(img.shape) == 2:
        img = img[np.newaxis, np.newaxis, :, :]
    elif len(img.shape) == 3:
        # Convert RGB to (B, C, H, W)
        if img.shape[2] == 3:
            img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :, :, :]
    
    # Convert to float and normalize
    img = img.astype(np.float32)
    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Convert to PyTorch tensor
    return torch.from_numpy(img)


def preprocess_tensor(input_path, input_type, vae_model=None, subfolder=None, device="cpu", **kwargs):
    """
    Unified preprocessing function for both images and VAE latents.
    
    Args:
        input_path (str): Path to image file
        input_type (str): Either 'image' or 'latent'
        vae_model (str, optional): VAE model name for latent processing
        subfolder (str, optional): Subfolder for VAE model
        device (str): Device to use
        **kwargs: Additional arguments for preprocessing functions
    
    Returns:
        torch.Tensor: Preprocessed tensor (image or VAE latent)
    """
    from .image_processing import load_image
    
    if input_type == "image":
        # Load and preprocess image directly
        img = load_image(input_path, grayscale=kwargs.get('grayscale', False))
        tensor = preprocess_image(
            img, 
            target_size=kwargs.get('target_size'),
            normalize=kwargs.get('normalize', True),
            force_power_of_two=kwargs.get('force_power_of_two', True),
            min_size=kwargs.get('min_size', 256)
        )
        return tensor.to(device)
    
    elif input_type == "latent":
        # Load image and encode to VAE latent
        if vae_model is None:
            raise ValueError("VAE model must be specified for latent processing")
        
        img = load_image(input_path, grayscale=kwargs.get('grayscale', False))
        
        # Load VAE model and processor
        vae, processor = load_vae_model(vae_model, subfolder)
        vae = vae.to(device)
        vae.eval()
        
        # Preprocess with VAE processor
        img_tensor = preprocess_image_with_vae_processor(img, processor)
        
        # Handle QwenImage specific processing
        from diffusers.models.autoencoders import AutoencoderKLQwenImage
        if isinstance(vae, AutoencoderKLQwenImage):
            img_tensor = img_tensor.unsqueeze(2)
        
        # Encode to latent space
        latent, _, _ = encode_image_to_latent(vae, processor, img_tensor, device)
        
        # Handle QwenImage output
        if isinstance(vae, AutoencoderKLQwenImage):
            latent = latent.squeeze(2)
        
        return latent
    
    else:
        raise ValueError(f"Unsupported input_type: {input_type}. Use 'image' or 'latent'.")
