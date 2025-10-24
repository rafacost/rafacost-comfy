# ComfyUI-DreamOmni2-GGUF

A **ComfyUI custom node** to run **DreamOmni2 GGUF** models directly inside ComfyUI — now with **multimodal (image + text)** support through the **llama-mtmd-cli** tool.

⚠️ **Work In Progress – use at your own risk.** <sub>(Or better: fork and help improve it.)</sub>

---

## ✨ Features

* Load and run **DreamOmni2 GGUF** models directly in ComfyUI.
* Supports **image + text multimodal pipelines** via `llama-mtmd-cli`.
* Accepts up to **two image inputs** from ComfyUI’s `Load Image` node.
* Returns **text output** or **conditioning embeddings**.
* Works with **quantized GGUF** models for high performance.

---

## 🧩 Prerequisites

You need the **`llama-mtmd-cli`** executable — it handles the image encoder and multimodal context for Qwen2VL and DreamOmni2-style models.

### Option 1 – Download Precompiled Binary

Download the latest prebuilt version from the official [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases).
Look for:

```
llama-mtmd-cli.exe
```

Place it anywhere and copy its full path. You’ll provide this path in the node.

### Option 2 – Build It Yourself

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

The compiled binary will appear at:

```
build/bin/Release/llama-mtmd-cli.exe
```

You can copy it into your ComfyUI `custom_nodes/rafacost-comfy/` folder or another location of your choice.

---

## 🧠 Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:

   ```bash
   git clone https://github.com/rafacost/rafacost-comfy.git
   ```

2. Download the **DreamOmni2** model, mmproj and LORA files from [Hugging Face](https://huggingface.co/rafacost/DreamOmni2-7.6B-GGUF).

3. Place them here:

   ```
   ComfyUI/models/unet/
   ├── DreamOmni2-Vlm-Model-7.6B-Q5_K.gguf
   ├── DreamOmni2-Vlm-Model-7.6B-f16-mmproj.gguf
   ├── flux1-kontext-dev-Q5_K.gguf
   ComfyUI/models/loras/
   ├── DreamOmni2-7.6B-Edit-Lora.safetensors
   ```

---

## 🧰 Usage

1. Launch ComfyUI.
2. Add the **DreamOmni2VLM** node from the **rafacostComfy/VLM** category.
3. Configure the node:

   * **CLI Path:** Path to your `llama-mtmd-cli.exe`.
   * **Model:** Select your `.gguf` model from the dropdown.
   * **MMProj Path:** Path to `mmproj.gguf` (vision adapter).
   * **Images:** Connect up to two image inputs.
   * **Prompt:** Type your text prompt (e.g., “Apply the style of image 1 on image 2”).
4. Run the workflow.

The node will execute the CLI, process the images, and return text results to both:

* The **ComfyUI output**, and
* Your **terminal console** (for debug/logs).

---

## 🧪 Example

Try workflow folder, or the image_result.png in examples.

---

## ⚙️ Troubleshooting

| Problem                                  | Possible Cause                                                               | Fix                                                                      |
| ---------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `CLI failed: not found`                  | Wrong path to `llama-mtmd-cli.exe`.                                          | Verify path and escape backslashes (`C:\\path\\to\\llama-mtmd-cli.exe`). |
| Model loads but crashes                  | Out-of-memory or incompatible quantization.                                  | Try a smaller GGUF quant (e.g. `Q4_K` instead of `Q8_0`).                |
| `'NoneType' object is not iterable`      | Missing mmproj file.                                                         | Download or copy `mmproj.gguf` into the same folder as the model.        |
| No output text                           | The model didn’t produce text tokens.                                        | Increase `max_tokens` (e.g. 512 or 1024).                                |
| Garbled CLI text                         | Locale issue.                                                                | Run ComfyUI with UTF-8 environment: `set PYTHONUTF8=1`.                  |
| Low Image Quality                        | Low Base Model GGUF quant                                                    | Try a higher GGUF quant for BaseModel (eg. use flux_kontext:`Q8_0`)      |
| Low Prompt Adherence                     | Low DreamOmni2 GGUF quant                                                    | Try a higher GGUF quant for DreamOmni2 (eg. use DreamOmni2-GGUF:`Q8_0`)  |

---

## 🗒️ Notes

* Requires **ComfyUI ≥ 0.3.66**.
* Tested on **Python 3.12** and **Windows 11**.
* GPU recommended for large GGUFs.
* Original model licensing applies — this node only provides an interface.

---

## 📜 License

Refer to the original model and `llama.cpp` licenses.
This node does not modify or supersede any upstream licensing.

---

