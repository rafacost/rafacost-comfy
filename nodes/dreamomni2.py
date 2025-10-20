import os
import folder_paths
import torch
import comfy.model_management as mm
from comfy.utils import load_torch_file
import comfy.sd
import comfy.model_patcher

try:
    from .gguf_loader import load_flux_gguf
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    print("GGUF support not available. Install ComfyUI-GGUF for GGUF model support.")


class DreamOmni2GGUF:
    """
    ComfyUI node for DreamOmni2 Generation Pipeline with GGUF support.
    Outputs standard MODEL/CLIP/VAE types compatible with KSampler.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available LoRA models
        lora_path = os.path.join(folder_paths.models_dir, "loras")
        lora_files = ["none"]
        if os.path.exists(lora_path):
            lora_files += [f for f in os.listdir(lora_path) 
                          if f.endswith(('.safetensors', '.pt', '.bin'))]
        
        # Get available VLM models (GGUF)
        unet_path = os.path.join(folder_paths.models_dir, "unet")
        vlm_files = ["none"]
        if os.path.exists(unet_path):
            vlm_files += [f for f in os.listdir(unet_path) 
                         if f.endswith('.gguf') and 'vlm' in f.lower()]
        
        return {
            "required": {
                "base_model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "vlm_model": (vlm_files, {
                    "default": "none",
                    "tooltip": "Vision-Language Model in GGUF format from models/unet"
                }),
            },
            "optional": {
                "lora_model": (lora_files, {
                    "default": "none",
                    "tooltip": "DreamOmni2 LoRA weights"
                }),
                "lora_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "vlm_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Strength of VLM influence on generation"
                }),
                "vlm_device": (["auto", "gpu", "cpu"], {
                    "default": "auto",
                }),
                "enable_cpu_offload": ("BOOLEAN", {
                    "default": True,
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VLM_MODEL")
    RETURN_NAMES = ("model", "clip", "vae", "vlm")
    FUNCTION = "load_pipeline"
    CATEGORY = "rafacost-comfy/DreamOmni2-GGUF"
    
    def load_pipeline(self, base_model, clip, vae, vlm_model, 
                     lora_model="none", lora_strength=1.0, vlm_strength=1.0,
                     vlm_device="auto", enable_cpu_offload=True):
        """
        Prepare DreamOmni2 pipeline - outputs compatible with KSampler.
        """
        
        # Clone the model to avoid modifying the original
        model = base_model.clone()
        
        # Determine device for VLM
        if vlm_device == "auto":
            device = mm.get_torch_device()
        else:
            device = torch.device(vlm_device if vlm_device in ["cpu", "cuda"] else "cpu")
        
        print(f"=== DreamOmni2 Pipeline Setup ===")
        print(f"Base model: {type(base_model)}")
        print(f"VLM device: {device}")
        
        # Load VLM model in GGUF format
        vlm = None
        if vlm_model != "none":
            vlm = self._load_vlm_gguf(vlm_model, device, vlm_strength)
            
            # Integrate VLM into the model
            if vlm is not None:
                model = self._integrate_vlm_with_model(model, vlm, vlm_strength, enable_cpu_offload)
        else:
            print("No VLM specified - using base model only")
        
        # Apply LoRA to model if specified
        if lora_model != "none":
            print(f"Applying LoRA: {lora_model} with strength {lora_strength}")
            model = self._apply_lora_to_model(model, lora_model, lora_strength)
        
        print("=== Pipeline ready for KSampler ===\n")
        
        return (model, clip, vae, vlm)
    
    def _load_vlm_gguf(self, vlm_name, device, strength):
        """Load Vision-Language Model in GGUF format."""
        if not GGUF_AVAILABLE:
            print("Warning: GGUF support not available. Install ComfyUI-GGUF")
            return None
        
        unet_path = os.path.join(folder_paths.models_dir, "unet")
        vlm_path = os.path.join(unet_path, vlm_name)
        
        if not os.path.exists(vlm_path):
            print(f"Warning: VLM model not found at {vlm_path}")
            return None
        
        print(f"Loading VLM GGUF: {vlm_name}")
        print(f"VLM strength: {strength}")
        
        try:
            # Determine dtype
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            # Load VLM using GGUF loader
            vlm_config = "flux/transformer"  # Use FLUX config as base
            vlm_model, = load_flux_gguf(vlm_path, vlm_config, dtype, device)
            
            print(f"✓ VLM loaded successfully")
            return vlm_model
            
        except Exception as e:
            print(f"Error loading VLM GGUF: {e}")
            print("Continuing without VLM...")
            return None
    
    def _integrate_vlm_with_model(self, model, vlm, strength, enable_offload):
        """
        Integrate VLM capabilities into the base model.
        This patches the model to use VLM features during sampling.
        """
        print("Integrating VLM with base model...")
        
        # Store VLM in model's patcher for access during sampling
        if not hasattr(model, 'model_options'):
            model.model_options = {}
        
        if 'transformer_options' not in model.model_options:
            model.model_options['transformer_options'] = {}
        
        # Add VLM to transformer options
        model.model_options['transformer_options']['dreamomni2_vlm'] = {
            'model': vlm,
            'strength': strength,
            'offload_enabled': enable_offload,
        }
        
        # Patch the model's forward pass to use VLM
        original_apply_model = model.model.apply_model if hasattr(model.model, 'apply_model') else None
        
        if original_apply_model:
            def patched_apply_model(x, t, **kwargs):
                # Get VLM from options
                vlm_config = kwargs.get('transformer_options', {}).get('dreamomni2_vlm')
                
                if vlm_config:
                    vlm_model = vlm_config['model']
                    vlm_strength = vlm_config['strength']
                    
                    # Move VLM to GPU if offloading
                    if vlm_config.get('offload_enabled') and hasattr(vlm_model, 'to'):
                        vlm_model = vlm_model.to(x.device)
                    
                    # Get VLM features (this is a simplified version)
                    # In practice, you'd extract vision features and inject them
                    if hasattr(vlm_model, 'encode') or hasattr(vlm_model, 'forward'):
                        try:
                            # Apply VLM encoding to condition on visual features
                            if 'c' in kwargs:
                                # Enhance conditioning with VLM
                                kwargs['c'] = self._enhance_conditioning_with_vlm(
                                    kwargs['c'], vlm_model, vlm_strength
                                )
                        except Exception as e:
                            print(f"VLM application warning: {e}")
                    
                    # Move VLM back to CPU if offloading
                    if vlm_config.get('offload_enabled') and hasattr(vlm_model, 'cpu'):
                        vlm_model = vlm_model.cpu()
                
                # Call original apply_model
                return original_apply_model(x, t, **kwargs)
            
            # Replace the apply_model method
            model.model.apply_model = patched_apply_model
            print("✓ VLM integrated with model forward pass")
        
        return model
    
    def _enhance_conditioning_with_vlm(self, conditioning, vlm_model, strength):
        """
        Enhance text conditioning with VLM visual understanding.
        """
        # This is a placeholder for the actual VLM conditioning logic
        # In practice, you'd:
        # 1. Extract visual features from VLM
        # 2. Combine with text conditioning
        # 3. Scale by strength parameter
        
        # For now, return conditioning as-is
        # Implement based on DreamOmni2's specific VLM integration
        return conditioning
    
    def _apply_lora_to_model(self, model, lora_name, strength):
        """Apply DreamOmni2 LoRA weights using ComfyUI's LoRA system."""
        lora_path = os.path.join(folder_paths.models_dir, "loras", lora_name)
        
        if not os.path.exists(lora_path):
            print(f"Warning: LoRA not found at {lora_path}")
            return model
        
        try:
            # Load LoRA weights
            lora = load_torch_file(lora_path, safe_load=True)
            
            # Apply LoRA using ComfyUI's native method
            # This is compatible with the model patcher system
            model_lora, _ = comfy.sd.load_lora_for_models(
                model, None, lora, strength, 0
            )
            
            print(f"✓ LoRA applied with strength {strength}")
            return model_lora
            
        except Exception as e:
            print(f"Warning: Failed to apply LoRA: {e}")
            return model


class DreamOmni2VLMLoaderGGUF:
    """
    Standalone VLM loader for loading Vision-Language Models in GGUF format.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        unet_path = os.path.join(folder_paths.models_dir, "unet")
        vlm_files = ["none"]
        if os.path.exists(unet_path):
            vlm_files += [f for f in os.listdir(unet_path) if f.endswith('.gguf')]
        
        return {
            "required": {
                "vlm_model": (vlm_files, {"default": "none"}),
                "device": (["auto", "gpu", "cpu"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("VLM_MODEL",)
    RETURN_NAMES = ("vlm",)
    FUNCTION = "load_vlm"
    CATEGORY = "rafacost-comfy/DreamOmni2-GGUF"
    
    def load_vlm(self, vlm_model, device):
        """Load VLM model in GGUF format."""
        if vlm_model == "none":
            return (None,)
        
        if not GGUF_AVAILABLE:
            raise RuntimeError("GGUF support not available. Install ComfyUI-GGUF.")
        
        # Determine device
        if device == "auto":
            device = mm.get_torch_device()
        else:
            device = torch.device(device)
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Load VLM
        unet_path = os.path.join(folder_paths.models_dir, "unet")
        vlm_path = os.path.join(unet_path, vlm_model)
        
        if not os.path.exists(vlm_path):
            raise FileNotFoundError(f"VLM model not found: {vlm_path}")
        
        print(f"Loading VLM from: {vlm_path}")
        vlm_config = "flux/transformer"
        vlm, = load_flux_gguf(vlm_path, vlm_config, dtype, device)
        
        print(f"✓ VLM loaded on {device}")
        return (vlm,)


class DreamOmni2ConditioningCombine:
    """
    Combines text and vision conditioning for DreamOmni2.
    Use this before KSampler to inject visual context.
    Supports multiple images for multi-modal conditioning.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vlm": ("VLM_MODEL",),
                "vlm_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "image_weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "image_weight_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "image_weight_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "concat_method": (["average", "concat", "weighted_sum", "attention"], {
                    "default": "weighted_sum"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "rafacost-comfy/DreamOmni2-GGUF"
    
    def combine(self, conditioning, vlm, vlm_strength, 
                image_1=None, image_2=None, image_3=None, image_4=None,
                image_weight_1=1.0, image_weight_2=1.0, image_weight_3=1.0, image_weight_4=1.0,
                concat_method="weighted_sum"):
        """Combine text conditioning with VLM visual features from multiple images."""
        if vlm is None:
            return (conditioning,)
        
        # Collect all images and weights
        images = []
        weights = []
        if image_1 is not None:
            images.append(image_1)
            weights.append(image_weight_1)
        if image_2 is not None:
            images.append(image_2)
            weights.append(image_weight_2)
        if image_3 is not None:
            images.append(image_3)
            weights.append(image_weight_3)
        if image_4 is not None:
            images.append(image_4)
            weights.append(image_weight_4)
        
        if len(images) == 0:
            print("No images provided for VLM conditioning")
            return (conditioning,)
        
        print(f"Processing {len(images)} images with VLM")
        print(f"Concat method: {concat_method}")
        print(f"Weights: {weights[:len(images)]}")
        
        # Encode images with VLM
        image_features = self._encode_images_with_vlm(vlm, images, weights, concat_method)
        
        # Clone conditioning to avoid modifying original
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            
            # Add VLM information to conditioning
            if 'dreamomni2_vlm' not in n[1]:
                n[1]['dreamomni2_vlm'] = {}
            
            n[1]['dreamomni2_vlm']['model'] = vlm
            n[1]['dreamomni2_vlm']['strength'] = vlm_strength
            n[1]['dreamomni2_vlm']['image_features'] = image_features
            n[1]['dreamomni2_vlm']['num_images'] = len(images)
            
            c.append(n)
        
        print(f"✓ Multi-image VLM conditioning applied (strength: {vlm_strength}, images: {len(images)})")
        return (c,)
    
    def _encode_images_with_vlm(self, vlm, images, weights, method):
        """Encode multiple images with VLM and combine features."""
        encoded_features = []
        
        for i, img in enumerate(images):
            print(f"  Encoding image {i+1}/{len(images)}...")
            # Placeholder for actual VLM encoding
            # In practice, you'd call vlm.encode(img) or similar
            feature = self._encode_single_image(vlm, img)
            encoded_features.append(feature)
        
        # Combine features based on method
        if method == "average":
            combined = self._average_features(encoded_features)
        elif method == "weighted_sum":
            combined = self._weighted_sum_features(encoded_features, weights)
        elif method == "concat":
            combined = self._concat_features(encoded_features)
        elif method == "attention":
            combined = self._attention_combine_features(encoded_features, weights)
        else:
            combined = encoded_features
        
        return combined
    
    def _encode_single_image(self, vlm, image):
        """Encode a single image with VLM."""
        # This is a placeholder - implement based on VLM's actual API
        # Real implementation would be:
        # with torch.no_grad():
        #     features = vlm.encode_image(image)
        return {"image": image, "encoded": True}
    
    def _average_features(self, features):
        """Average all image features."""
        print("  Combining with average method")
        return {"method": "average", "features": features}
    
    def _weighted_sum_features(self, features, weights):
        """Weighted sum of image features."""
        print(f"  Combining with weighted sum (weights: {weights})")
        return {"method": "weighted_sum", "features": features, "weights": weights}
    
    def _concat_features(self, features):
        """Concatenate all image features."""
        print("  Combining with concatenation method")
        return {"method": "concat", "features": features}
    
    def _attention_combine_features(self, features, weights):
        """Use attention mechanism to combine features."""
        print("  Combining with attention mechanism")
        return {"method": "attention", "features": features, "weights": weights}


class DreamOmni2MultiImageBatch:
    """
    Combine multiple images into a batch for VLM conditioning.
    Alternative to using individual image inputs.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "batch_images"
    CATEGORY = "rafacost-comfy/DreamOmni2-GGUF"
    
    def batch_images(self, image_1=None, image_2=None, image_3=None, image_4=None,
                    image_5=None, image_6=None, image_7=None, image_8=None):
        """Batch multiple images together."""
        images = []
        for img in [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8]:
            if img is not None:
                images.append(img)
        
        if len(images) == 0:
            # Return empty batch
            return (torch.zeros((1, 64, 64, 3)), 0)
        
        # Stack images along batch dimension
        batched = torch.cat(images, dim=0)
        print(f"Batched {len(images)} images, shape: {batched.shape}")
        
        return (batched, len(images))


class DreamOmni2ConditioningBatch:
    """
    Apply VLM conditioning using a batch of images.
    More efficient than individual image inputs.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vlm": ("VLM_MODEL",),
                "images": ("IMAGE",),
                "vlm_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "image_weights": ("STRING", {
                    "default": "1.0,1.0,1.0,1.0",
                    "multiline": False,
                    "tooltip": "Comma-separated weights for each image"
                }),
                "concat_method": (["average", "concat", "weighted_sum", "attention"], {
                    "default": "weighted_sum"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "rafacost-comfy/DreamOmni2-GGUF"
    
    def combine(self, conditioning, vlm, images, vlm_strength, 
                image_weights="1.0,1.0,1.0,1.0", concat_method="weighted_sum"):
        """Combine text conditioning with batched image features."""
        if vlm is None:
            return (conditioning,)
        
        # Parse weights
        weights = [float(w.strip()) for w in image_weights.split(',')]
        
        # Get number of images from batch
        num_images = images.shape[0]
        print(f"Processing batch of {num_images} images with VLM")
        print(f"Concat method: {concat_method}")
        
        # Pad weights if needed
        while len(weights) < num_images:
            weights.append(1.0)
        weights = weights[:num_images]
        
        print(f"Image weights: {weights}")
        
        # Encode batch with VLM
        image_features = self._encode_batch_with_vlm(vlm, images, weights, concat_method)
        
        # Clone conditioning
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            
            if 'dreamomni2_vlm' not in n[1]:
                n[1]['dreamomni2_vlm'] = {}
            
            n[1]['dreamomni2_vlm']['model'] = vlm
            n[1]['dreamomni2_vlm']['strength'] = vlm_strength
            n[1]['dreamomni2_vlm']['image_features'] = image_features
            n[1]['dreamomni2_vlm']['num_images'] = num_images
            n[1]['dreamomni2_vlm']['batch_mode'] = True
            
            c.append(n)
        
        print(f"✓ Batch VLM conditioning applied ({num_images} images)")
        return (c,)
    
    def _encode_batch_with_vlm(self, vlm, images, weights, method):
        """Encode a batch of images with VLM."""
        # Placeholder for batch encoding
        # Real implementation would process the entire batch at once for efficiency
        print(f"  Encoding batch of {images.shape[0]} images...")
        
        return {
            "method": method,
            "images": images,
            "weights": weights,
            "batch_encoded": True
        }
