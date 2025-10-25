import subprocess
import os
import folder_paths
import torch
from PIL import Image
import numpy as np



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

class DreamOmni2VLM():
    @classmethod
    def INPUT_TYPES(cls):
        # 获取所有配置的 unet 模型路径（包括默认路径和 extra_model_paths 中的路径）
        unet_paths = folder_paths.get_folder_paths("unet")
        print(f"[DreamOmni2-VLM] 已配置的 unet 模型路径（包括 extra_model_paths）：", flush=True)
        for i, path in enumerate(unet_paths, 1):
            print(f"  {i}. {path}", flush=True)
        
        # 收集所有路径下的 .gguf 模型和 mmproj 文件
        ggufs = ["none"]
        mmprojs = ["none"]
        seen_ggufs = set()  # 避免重复文件（不同路径可能有同名文件）
        seen_mmprojs = set()

        for unet_dir in unet_paths:
            if os.path.isdir(unet_dir):
                print(f"\n[DreamOmni2-VLM] 扫描路径: {unet_dir}", flush=True)
                # 收集所有 .gguf 文件
                for f in os.listdir(unet_dir):
                    if f.lower().endswith(".gguf"):
                        if f not in seen_ggufs:
                            seen_ggufs.add(f)
                            ggufs.append(f)
                            print(f"  发现模型: {f}", flush=True)
                        # 同时检查是否为 mmproj 文件
                        if "mmproj" in f.lower() and f not in seen_mmprojs:
                            seen_mmprojs.add(f)
                            mmprojs.append(f)
                            print(f"  发现 mmproj: {f}", flush=True)
            else:
                print(f"\n[DreamOmni2-VLM] 路径不存在: {unet_dir}", flush=True)

        print(f"\n[DreamOmni2-VLM] 共发现 {len(ggufs)-1} 个模型，{len(mmprojs)-1} 个 mmproj 文件", flush=True)

        return {
            "required": {
                "cli_path": ("STRING", {"default": "C:\\path\\to\\llama-mtmd-cli.exe"}),
                "model_name": (ggufs, {"default": ggufs[0] if ggufs else ""}),
                "mmproj_path": (mmprojs, {"default": mmprojs[0] if mmprojs else ""}),
                "prompt": ("STRING", {"default": "Describe the image(s).", "multiline": True}),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "clip": ("CLIP",),  # from Flux DualCLIPLoader
                "temperature": ("FLOAT", {"default": 0.7}),
                "max_tokens": ("INT", {"default": 1024}),
                "as_conditioning": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "text")
    FUNCTION = "run"
    CATEGORY = "rafacostComfy/VLM"

    def run(self, cli_path, model_name, mmproj_path, prompt,
            image1, image2, clip,temperature, max_tokens, as_conditioning):
        try:
            # 方法入口打印
            print("\n" + "="*50, flush=True)
            print("[DreamOmni2-VLM] 开始执行run方法", flush=True)
            print("="*50 + "\n", flush=True)

            # 处理模型路径（自动包含 extra_model_paths 中的路径）
            model_full_path = folder_paths.get_full_path("unet", model_name)
            print(f"[DreamOmni2-VLM] 模型名称: {model_name}", flush=True)
            print(f"[DreamOmni2-VLM] 模型完整路径: {model_full_path}", flush=True)
            
            if model_name != "none":
                if os.path.exists(model_full_path):
                    file_size = os.path.getsize(model_full_path) / (1024**3)  # GB
                    print(f"[DreamOmni2-VLM] 模型存在，大小: {file_size:.2f} GB", flush=True)
                else:
                    print(f"[DreamOmni2-VLM] 错误：模型文件不存在！", flush=True)
            else:
                print(f"[DreamOmni2-VLM] 未选择模型", flush=True)

            # 处理mmproj路径（自动包含 extra_model_paths 中的路径）
            mmproj_full_path = folder_paths.get_full_path("unet", mmproj_path)
            print(f"\n[DreamOmni2-VLM] mmproj名称: {mmproj_path}", flush=True)
            print(f"[DreamOmni2-VLM] mmproj完整路径: {mmproj_full_path}", flush=True)
            
            if mmproj_path != "none":
                if os.path.exists(mmproj_full_path):
                    file_size = os.path.getsize(mmproj_full_path) / (1024**2)  # MB
                    print(f"[DreamOmni2-VLM] mmproj存在，大小: {file_size:.2f} MB", flush=True)
                else:
                    print(f"[DreamOmni2-VLM] 错误：mmproj文件不存在！", flush=True)
            else:
                print(f"[DreamOmni2-VLM] 未选择mmproj", flush=True)

            # 处理图像临时文件
            tmp1 = os.path.join(folder_paths.get_temp_directory(), "vlm_img1.png")
            tmp2 = os.path.join(folder_paths.get_temp_directory(), "vlm_img2.png")
            print(f"\n[DreamOmni2-VLM] 临时图像路径1: {tmp1}", flush=True)
            print(f"[DreamOmni2-VLM] 临时图像路径2: {tmp2}", flush=True)

            # 保存图像
            try:
                img1 = tensor_to_pil(image1)
                img1.save(tmp1)
                print(f"[DreamOmni2-VLM] 图像1保存成功", flush=True)
                
                img2 = tensor_to_pil(image2)
                img2.save(tmp2)
                print(f"[DreamOmni2-VLM] 图像2保存成功", flush=True)
            except Exception as e:
                print(f"[DreamOmni2-VLM] 图像保存失败: {str(e)}", flush=True)
                raise

            # 打印提示词
            print(f"\n[DreamOmni2-VLM] 提示词: {prompt}", flush=True)

            # 构建命令
            cmd = [
                cli_path,
                "--model", model_full_path,
                "--mmproj", mmproj_full_path,
                "--temp", str(temperature),
                "--n-predict", str(max_tokens),
                "--image", tmp1,
                "--image", tmp2,
                "--prompt", prompt,
            ]
            print(f"\n[DreamOmni2-VLM] 执行命令: {' '.join(cmd)}", flush=True)

            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(f"\n[DreamOmni2-VLM] 命令返回码: {result.returncode}", flush=True)

            if result.returncode != 0:
                print(f"[DreamOmni2-VLM] 命令错误输出: {result.stderr}", flush=True)
                raise RuntimeError(f"CLI failed: {result.stderr.strip()}")

            # 处理输出
            output = result.stdout.strip()
            print("\n" + "="*50, flush=True)
            print(f"[DreamOmni2-VLM] 模型输出:\n{output}", flush=True)
            print("="*50 + "\n", flush=True)

            if as_conditioning:
                clip_text = output.strip()
                tokens = clip.tokenize(clip_text)
                embedding, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

                # 确保是tensor类型
                embedding = torch.tensor(embedding) if not isinstance(embedding, torch.Tensor) else embedding
                pooled = torch.tensor(pooled) if not isinstance(pooled, torch.Tensor) else pooled

                conditioning = [[embedding, {"clip": embedding, "pooled_output": pooled, "text": clip_text}]]

                print(f"[DreamOmni2-VLM] 返回conditioning数据，长度: {len(conditioning)}", flush=True)
                return (conditioning, clip_text)
            else:
                print(f"[DreamOmni2-VLM] 返回文本输出，长度: {len(output)}", flush=True)
                return (None, output)

        except Exception as e:
            print(f"\n[DreamOmni2-VLM] 执行出错: {str(e)}", flush=True)
            raise  # 重新抛出异常不影响原有错误处理

NODE_CLASS_MAPPINGS = {"DreamOmni2-VLM": DreamOmni2VLM}
NODE_DISPLAY_NAME_MAPPINGS = {"DreamOmni2-VLM": "DreamOmni2VLM"}