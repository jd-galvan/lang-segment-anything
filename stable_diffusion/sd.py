import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline



class SD:
    def __init__(self, sd_model="runwayml/stable-diffusion-inpainting", device='cuda'):
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            sd_model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipeline = pipeline.to("cuda")


    def predict(
        self,
        image_path,
        mask_path
    ):
        init_image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")

        new_image = self.pipeline(prompt="Impaint the image.", image=init_image, mask_image=mask_image).images[0]

        return new_image
