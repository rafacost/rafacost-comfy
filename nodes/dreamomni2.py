import os
import re
import torch
import numpy as np
import folder_paths
import hashlib
import base64
import json
from PIL import Image
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler


def tensor_to_pil(t):
    if t.ndim == 4:  # (B, C, H, W)
        t = t[0]
    elif t.ndim == 5:  # (1, 1, H, W, C)
        t = t[0, 0]
    if t.shape[0] in (1, 3):  # (C, H, W)
        t = t.permute(1, 2, 0)
    arr = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


class DreamOmni2VLM:
    CACHEABLE = True
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        extra = folder_paths.get_folder_paths("unet")
        ggufs, mmprojs = ["none"], ["none"]
        for path in extra:
            if os.path.isdir(path):
                ggufs += [f for f in os.listdir(path) if f.lower().endswith(".gguf")]
                mmprojs += [f for f in os.listdir(path) if f.lower().endswith(".gguf") and "mmproj" in f.lower()]

        return {
            "required": {
                "model_name": (ggufs, {"default": ggufs[0]}),
                "mmproj_path": (mmprojs, {"default": mmprojs[0]}),
                "prompt": ("STRING", {"default": "Describe the images with detail.", "multiline": True}),
                "clip": ("CLIP",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "seed": ("INT", {"default": 42}),
                "temperature": ("FLOAT", {"default": 0.5}),
                "max_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "n_ctx": ("INT", {"default": 2048, "min": 0, "max": 128000, "step": 256}),
                "as_conditioning": ("BOOLEAN", {"default": True}),
                "use_cache": ("BOOLEAN", {"default": True}),
            },
            "optional": {                
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
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

    def _generate_cache_key(self, model_name, mmproj_path, prompt, image1, image2, image3, image4, seed, temperature, max_tokens, n_ctx):
        img_hash = lambda t: hashlib.md5(t.cpu().numpy().tobytes()).hexdigest()
        key = f"{model_name}_{mmproj_path}_{prompt}_{img_hash(image1)}_{img_hash(image2)}_{img_hash(image3)}_{img_hash(image4)}_{seed}_{temperature}_{max_tokens}"
        return hashlib.md5(key.encode()).hexdigest()

    def run(self, model_name, mmproj_path, prompt, image1, image2, clip,
            temperature, max_tokens, as_conditioning, seed, n_ctx, use_cache=True, 
            unique_id=None, extra_pnginfo=None, image3=None, image4=None):

        # Avoid using Python's `or` with tensors (raises: "Boolean value of Tensor with more than one value is ambiguous").
        # Use explicit None checks so we don't attempt to evaluate tensor truthiness.
        cache_key = self._generate_cache_key(
            model_name,
            mmproj_path,
            prompt,
            image1,
            image2,
            image3 if image3 is not None else image1,
            image4 if image4 is not None else image1,
            seed,
            temperature,
            max_tokens,
            n_ctx,
        )

        if use_cache and cache_key in self._cache:
            print(f"[rafacostComfy: DreamOmni2-VLM] Using cached result for seed {seed}")
            output = self._cache[cache_key]
        else:
            model_path = folder_paths.get_full_path("unet", model_name)
            mmproj_path = folder_paths.get_full_path("unet", mmproj_path)

            # Create list of available images, using None as placeholder for optional ones
            images = [image1, image2, image3, image4]
            # Filter out None values
            images = [img for img in images if img is not None]

            tmp_paths = []
            for i, img_t in enumerate(images, 1):
                tmp = os.path.join(folder_paths.get_temp_directory(), f"vlm_img{i}.png")
                tensor_to_pil(img_t).save(tmp)
                tmp_paths.append(tmp)

            images_data = [image_to_base64_data_uri(p) for p in tmp_paths]
            prompt = re.sub(r"[\"'`´]", "", prompt)

            # Initialize llama_cpp model with Qwen2.5-VL handler
            chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path)
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=-1,
                verbose=False,
                seed=seed,
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant describing images in detail."},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                              + [{"type": "image_url", "image_url": {"url": img}} for img in images_data],
                },
            ]

            result = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            #print(f"[rafacostComfy: DreamOmni2-VLM] Message:\n{messages}")
            output = result["choices"][0]["message"]["content"].strip()
            #print(f"[rafacostComfy: DreamOmni2-VLM] Result:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
            print(f"[rafacostComfy: DreamOmni2-VLM] Output:\n{output}")

            if use_cache:
                self._cache[cache_key] = output

        # auto-update UI
        if unique_id and extra_pnginfo:
            workflow = extra_pnginfo.get("workflow")
            if workflow and "nodes" in workflow:
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
                if node and "widgets_values" in node and len(node["widgets_values"]) > 3:
                    node["widgets_values"][3] = output

        if as_conditioning:
            clip_text = output.strip()
            tokens = clip.tokenize(clip_text)
            embedding, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            embedding = torch.tensor(embedding) if not isinstance(embedding, torch.Tensor) else embedding
            pooled = torch.tensor(pooled) if not isinstance(pooled, torch.Tensor) else pooled
            conditioning = [[embedding, {"clip": embedding, "pooled_output": pooled, "text": clip_text}]]
            return (conditioning, clip_text)
        else:
            return (None, output)


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


anytype = AnyType("*")


class DreamOmni2_Output_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"text": (anytype, {})},
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    CATEGORY = "rafacostComfy/VLM"
    FUNCTION = "main"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    def main(self, text, unique_id=None, extra_pnginfo=None):
        text_str = str(text)
        if unique_id and extra_pnginfo and len(extra_pnginfo) > 0:
            workflow = extra_pnginfo.get("workflow")
            if workflow and "nodes" in workflow:
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
                if node:
                    node["widgets_values"] = [text_str]
        return {"ui": {"text": [text_str]}}


NODE_CLASS_MAPPINGS = {
    "DreamOmni2-VLM": DreamOmni2VLM,
    "DreamOmni2_Output_Node": DreamOmni2_Output_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamOmni2-VLM": "DreamOmni2 - VLM",
    "DreamOmni2_Output_Node": "DreamOmni2 Output Node",
}
