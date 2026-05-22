# -*- coding: utf-8 -*-
"""AI Text-to-Image Generator - SDXL for Hugging Face Spaces"""

import torch
import gradio as gr
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler
from PIL import Image
import gc
import random
import tempfile
import os

# -------------------------------
# 1. Device & Precision Setup
# -------------------------------
runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
precision_type = torch.float16 if runtime_device == "cuda" else torch.float32
print(f"Running on: {runtime_device} | Data type: {precision_type}")

# -------------------------------
# 2. Custom VAE Loader
# -------------------------------
def load_stable_vae(location="madebyollin/sdxl-vae-fp16-fix"):
    vae_component = AutoencoderKL.from_pretrained(
        location,
        torch_dtype=precision_type
    )
    return vae_component.to("cpu")

print("Acquiring VAE encoder...")
vae_encoder = load_stable_vae()

# -------------------------------
# 3. Core Pipeline Builder
# -------------------------------
base_repo = "stabilityai/stable-diffusion-xl-base-1.0"

print("Assembling generation pipeline...")
pipeline_core = StableDiffusionXLPipeline.from_pretrained(
    base_repo,
    vae=vae_encoder,
    torch_dtype=precision_type,
    use_safetensors=True,
    variant="fp16" if runtime_device == "cuda" else None,
    safety_checker=None,
    requires_safety_checker=False
)

# Replace default scheduler with DPMSolver for speed/quality
pipeline_core.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline_core.scheduler.config,
    use_karras_sigmas=True,
    algorithm_type="sde-dpmsolver++"
)

# -------------------------------
# 4. Memory Offloading (VRAM saver)
# -------------------------------
if runtime_device == "cuda":
    pipeline_core.enable_model_cpu_offload()
    pipeline_core.enable_attention_slicing()
    print("Memory offloading activated (GPU RAM saver).")
else:
    pipeline_core = pipeline_core.to(runtime_device)

print("All components ready.\n")

# -------------------------------
# 5. Image Generation Function
# -------------------------------
def create_image_from_text(
    user_prompt: str,
    unwanted_content: str = "blurry, ugly, low quality, distorted, bad anatomy, watermark",
    iteration_count: int = 28,
    creativity_level: float = 7.2,
    output_width: int = 1024,
    output_height: int = 1024,
    random_seed: int = None
) -> str:
    """
    Produces a high‑resolution image, saves it as a PNG file,
    and returns the file path for Gradio to serve.
    """
    # Seed
    if random_seed is None:
        random_seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=runtime_device).manual_seed(random_seed)
    print(f"Generating with seed {random_seed}: '{user_prompt[:60]}...'")

    # Clean up memory before run
    if runtime_device == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    try:
        with torch.inference_mode():
            result_image = pipeline_core(
                prompt=user_prompt,
                negative_prompt=unwanted_content,
                num_inference_steps=iteration_count,
                guidance_scale=creativity_level,
                width=output_width,
                height=output_height,
                guidance_rescale=0.7,
                generator=generator
            ).images[0]
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print("VRAM limit reached – retrying with 768x768 resolution...")
            torch.cuda.empty_cache()
            gc.collect()
            result_image = pipeline_core(
                prompt=user_prompt,
                negative_prompt=unwanted_content,
                num_inference_steps=iteration_count,
                guidance_scale=creativity_level,
                width=768,
                height=768,
                guidance_rescale=0.7,
                generator=generator
            ).images[0]
        else:
            raise error

    if result_image.mode != "RGB":
        result_image = result_image.convert("RGB")

    # Save as PNG file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    result_image.save(temp_file.name, format="PNG")
    temp_file.close()

    file_kb = os.path.getsize(temp_file.name) / 1024
    print(f"PNG saved at: {temp_file.name} | size: {file_kb:.1f} KB")
    return temp_file.name

# -------------------------------
# 6. Gradio Interface
# -------------------------------
interface = gr.Interface(
    fn=create_image_from_text,
    inputs=[
        gr.Textbox(label="What do you want to see?", placeholder="Describe your image...", value=demo_examples[0][0]),
        gr.Textbox(label="Avoid these elements", value="blurry, ugly, low quality, distorted, bad anatomy, watermark"),
        gr.Slider(minimum=15, maximum=50, step=1, label="Sampling iterations", value=28),
        gr.Slider(minimum=2.0, maximum=15.0, step=0.5, label="Creative freedom (guidance)", value=7.2),
        gr.Slider(minimum=512, maximum=1024, step=64, label="Output width", value=1024),
        gr.Slider(minimum=512, maximum=1024, step=64, label="Output height", value=1024),
        gr.Number(label="Random seed (optional – leave empty for random)", value=None)
    ],
    outputs=gr.Image(label="Generated artwork (PNG)", type="filepath"),
    title="AI Text‑to‑Image Generator (SDXL)",
    description="Original SDXL Image Forge – High‑Resolution Text‑to‑Image Generator",
    examples=demo_examples   # now each example has all 7 values
)

# -------------------------------
# 5. Launch for Hugging Face Spaces
# -------------------------------
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
