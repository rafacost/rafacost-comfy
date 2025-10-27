# Changelog

## [0.4.0] - 2025-10-25
### Major Changes
- **Removed external CLI dependency**  
  DreamOmni2 now runs natively via `llama-cpp-python`, eliminating the need for `llama-mtmd-cli.exe`.
- **Full CUDA acceleration support**  
  Uses `llama-cpp-python` compiled with CUDA (FP16 / Q4 / Q8 quantization).
- **Adopted Qwen2.5-VLChatHandler**  
  Enables multi-image inference directly from Python.
- **Improved performance**  
  3–5× faster inference, zero subprocess overhead.
- **Cleaner workflow integration**  
  Maintains prompt auto-updates, cache support, and output nodes.
- **Reduced dependency footprint**  
  `requirements.txt` now only includes:
  ```txt
  torch>=2.2.0
  numpy
  Pillow
  llama-cpp-python>=0.3.2
