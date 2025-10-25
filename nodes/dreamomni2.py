import os
import subprocess
import re
import torch
import numpy as np
import folder_paths



class DreamOmni2VLM():
    CACHEABLE = True
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
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 1024, "min": 256, "max": 16384, "step": 256}),
                "as_conditioning": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "text")
    FUNCTION = "run"
    CATEGORY = "rafacostComfy/VLM"



    def run(self, cli_path, model_name, mmproj_path, prompt,
            image1, image2, clip,temperature, max_tokens, as_conditioning):
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
            "--temp", str(temperature),
            "--n-predict", str(max_tokens),
            "--image", tmp1,
            "--image", tmp2,
            "--prompt", "'" + prompt + "'",
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


NODE_CLASS_MAPPINGS = {"DreamOmni2-VLM": DreamOmni2VLM}
NODE_DISPLAY_NAME_MAPPINGS = {"DreamOmni2-VLM": "DreamOmni2VLM"}
