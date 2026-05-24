# AI_Art_Generator
A clean, memory-efficient Stable Diffusion XL pipeline built with Gradio. Features DPMSolver++ scheduler, automatic GPU offloading, resolution fallback on OOM, and high‑quality PNG output.


#  SDXL Image Forge – AI Art Generator

**Original implementation** – fully customised Stable Diffusion XL pipeline with Gradio UI. Generate stunning high‑resolution images from text prompts, save them as PNG files, and enjoy intelligent memory management.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

##  Features

- **State‑of‑the‑art model** – Uses `stabilityai/stable-diffusion-xl-base-1.0` with a fixed VAE (`madebyollin/sdxl-vae-fp16-fix`).
- **Fast scheduling** – DPMSolver++ scheduler (Karras sigmas, SDE‑DPMSolver++) – needs only 15‑30 steps.
- **Memory efficient** – `enable_model_cpu_offload()` + attention slicing; runs on 8‑12 GB VRAM.
- **Automatic fallback** – If 1024x1024 OOMs, retries at 768x768 automatically.
- **Reproducible seeds** – Optional manual seed input; random by default.
- **PNG output** – Every generated image is saved as a lossless PNG file (no compression artifacts).
- **Interactive UI** – Built with Gradio (includes sliders, examples, and custom styling).

---

Project Live Demo

Project live screen recording:

https://github.com/user-attachments/assets/48b3bf89-fbc7-4537-8b98-9a886a9aa83d


## Requirements

- Python 3.9+
- NVIDIA GPU with CUDA (recommended) – works on CPU but will be very slow.
- At least **8 GB VRAM** for 1024×1024 (12 GB recommended).

Install dependencies:

```bash

pip install torch diffusers gradio transformers accelerate Pillow

For CUDA support, install torch with the appropriate index (e.g., pip3 install torch --index-url https://download.pytorch.org/whl/cu118).

