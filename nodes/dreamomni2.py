import os
import subprocess
import re
import torch
import numpy as np
import folder_paths
import hashlib

class DreamOmni2VLM():
    CACHEABLE = True
    
    # Class-level cache for results
    _cache = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        # discover local unet models
        extra = folder_paths.get_folder_paths("unet") 
        ggufs = ["none"]
        mmprojs = ["none"]
        for path in extra:
            if os.path.isdir(path):
                ggufs += [f for f in os.listdir(path) if f.lower().endswith(".gguf")]
                mmprojs += [f for f in os.listdir(path) if f.lower().endswith(".gguf") and "mmproj" in f.lower()]
    
        return {
            "required": {
                "cli_path": ("STRING", {"default": "C:\\path\\to\\llama-mtmd-cli.exe"}),
                "model_name": (ggufs, {"default": ggufs[0] if ggufs else ""}),
                "mmproj_path": (mmprojs, {"default": mmprojs[0] if mmprojs else ""}),
                "prompt": ("STRING", {"default": "Describe the images with detail.", "multiline": True}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "clip": ("CLIP",), 
                "seed": ("INT", {"default": 42}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 1024, "min": 256, "max": 16384, "step": 256}),
                "as_conditioning": ("BOOLEAN", {"default": True}),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "text")
    FUNCTION = "run"
    CATEGORY = "rafacostComfy/VLM"

    def _generate_cache_key(self, model_name, mmproj_path, prompt, image1, image2, seed, temperature, max_tokens):
        """Generate a unique cache key based on inputs"""
        # Create hash of image tensors
        img1_hash = hashlib.md5(image1.cpu().numpy().tobytes()).hexdigest()
        img2_hash = hashlib.md5(image2.cpu().numpy().tobytes()).hexdigest()
        
        # Combine all parameters into cache key
        key_string = f"{model_name}_{mmproj_path}_{prompt}_{img1_hash}_{img2_hash}_{seed}_{temperature}_{max_tokens}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def run(self, cli_path, model_name, mmproj_path, prompt,
            image1, image2, clip, temperature, max_tokens, as_conditioning, seed, 
            use_cache=True, unique_id=None, extra_pnginfo=None):
        
        # Generate cache key
        cache_key = self._generate_cache_key(model_name, mmproj_path, prompt, image1, image2, seed, temperature, max_tokens)
        
        # Check cache if enabled
        if use_cache and cache_key in self._cache:
            print(f"[rafacostComfy: DreamOmni2-VLM] Using cached result for seed {seed}")
            output = self._cache[cache_key]
        else:
            # Run inference
            model_path = folder_paths.get_full_path("unet", model_name)
            mmproj_path = folder_paths.get_full_path("unet", mmproj_path)
            
            from PIL import Image

            def tensor_to_pil(t):
                # Remove extra batch or mask dimensions
                if t.ndim == 4:  # (B, C, H, W)
                    t = t[0]
                elif t.ndim == 5:  # (1, 1, H, W, C)
                    t = t[0, 0]
                
                # Move channels last if needed
                if t.shape[0] in (1, 3):  # (C, H, W)
                    t = t.permute(1, 2, 0)

                arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
                return Image.fromarray(arr)

            # Save the images to temporary files
            tmp1 = os.path.join(folder_paths.get_temp_directory(), "vlm_img1.png")
            tmp2 = os.path.join(folder_paths.get_temp_directory(), "vlm_img2.png")

            img1 = tensor_to_pil(image1)
            img1.save(tmp1)

            img2 = tensor_to_pil(image2)
            img2.save(tmp2)

            # Clean prompt of problematic characters
            prompt = re.sub(r"[\"'`´]", "", prompt)  

            cmd = [
                cli_path,
                "--model", model_path,
                "--mmproj", mmproj_path,
                "--seed", str(seed),
                "--main-gpu", str(0),
                "--temp", str(temperature),
                "--n-predict", str(max_tokens),
                "--image", tmp1,
                "--image", tmp2,
                "--prompt", prompt,
            ]

            print("=================================================================")
            print(f"[rafacostComfy: DreamOmni2-VLM] Running LLAMA-MTMD-CLI: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"CLI failed: {result.stderr.strip()}")

            output = result.stdout.strip()
            output = re.findall(r"<gen>(.*?)</gen>", output, flags=re.DOTALL)
            output = output[0].strip() if output else ""
            
            print(f"[rafacostComfy: DreamOmni2-VLM] Output:\n{output}")
            print("=================================================================")
            
            # Store in cache
            if use_cache:
                self._cache[cache_key] = output

        # Update workflow widget with generated output
        if unique_id is not None and extra_pnginfo is not None:
            workflow = extra_pnginfo.get("workflow")
            if workflow and "nodes" in workflow:
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
                if node and "widgets_values" in node:
                    # Update the prompt widget (index 3 in the widgets list)
                    if len(node["widgets_values"]) > 3:
                        node["widgets_values"][3] = output

        if as_conditioning:
            clip_text = output.strip()
            tokens = clip.tokenize(clip_text)
            embedding, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

            # Ensure both are torch tensors
            embedding = torch.tensor(embedding) if not isinstance(embedding, torch.Tensor) else embedding
            pooled = torch.tensor(pooled) if not isinstance(pooled, torch.Tensor) else pooled
            conditioning = [[embedding, {"clip": embedding, "pooled_output": pooled, "text": clip_text}]]

            return (conditioning, clip_text)
        else:
            return (None, output)


class AnyType(str):
    """A special class that is always equal in not equal comparisons."""
    def __ne__(self, __value: object) -> bool:
        return False


anytype = AnyType("*")


class DreamOmni2_Output_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (anytype, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "rafacostComfy/VLM"
    FUNCTION = "main"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    def main(self, text, unique_id=None, extra_pnginfo=None):
        # Store the text value for the UI
        text_str = str(text)
        
        if unique_id is not None and extra_pnginfo is not None and len(extra_pnginfo) > 0:
            workflow = None
            if "workflow" in extra_pnginfo:
                workflow = extra_pnginfo["workflow"]
            node = None
            if workflow and "nodes" in workflow:
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                node["widgets_values"] = [text_str]
        
        # Return with proper format - text as a tuple with single element
        return {"ui": {"text": [text_str]}}


NODE_CLASS_MAPPINGS = {
    "DreamOmni2-VLM": DreamOmni2VLM,
    "DreamOmni2_Output_Node": DreamOmni2_Output_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamOmni2-VLM": "DreamOmni2 - VLM",
    "DreamOmni2_Output_Node": "DreamOmni2 Output Node",
}