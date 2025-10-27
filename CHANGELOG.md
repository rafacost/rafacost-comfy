# Changelog

## [0.4.1] - 2025-10-27
### Minor Changes
- **Fixed using only two or three images**  
  DreamOmni2 now can run with 2 to 4 images.

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
  ```

### Notes

* CLI-based versions are now deprecated (`v0.1.0–0.3.0`).
* The node remains backward-compatible with existing ComfyUI workflows.
* Recommended: install the CUDA build of `llama-cpp-python`:

  ```bash
  pip install -U llama-cpp-python --prefer-binary
  ```

  or for CUDA 12.4:

  ```bash
  pip install llama_cpp_python-0.3.2+cu124-cp312-cp312-win_amd64.whl
  ```

---

## [0.3.0] - 2025-10-20

* Added local cache system for seed-based reproducibility.
* Added prompt-output node for workflow display.
* Added support for four input images.

## [0.2.0] - 2025-10-15

* Added ComfyUI-level caching support.
* Added support for `extra_model_paths.yaml`.

## [0.1.0] - 2025-10-10

* Initial release with CLI-based DreamOmni2 GGUF inference.