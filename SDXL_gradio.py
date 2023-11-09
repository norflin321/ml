# pip install -U git+https://github.com/huggingface/diffusers.git controlnet_aux==0.0.7 > install_logs.txt
# pip install transformers accelerate safetensors mediapipe invisible_watermark gradio > install_logs.txt

import gradio as gr
from os import listdir
from os.path import isfile, join
import time
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL, StableDiffusionInstructPix2PixPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.pidi import PidiNetDetector
import torch
from matplotlib import pyplot as plt
import datetime
import os
import torch
import gc

euler_a = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch.float16, varient="fp16").to("cuda")
pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

results_url = "C:/Users/user/Desktop/ML/SDXL_results"
finetunes_url = "C:/Users/user/Desktop/ML/finetune_results"
finetune_choices = [f for f in listdir(finetunes_url) if isfile(join(finetunes_url, f))]
finetune_choices.insert(0, " ")
default_prompt = "a magic creature in style of sks, 3D, blender, perfect, 4k graphics, highly detailed, cute, pretty"
default_negative_prompt = "extra digit, fewer digits, cropped, worst quality, low quality, glitch, deformed, mutated, ugly, disfigured"

pipe = None
loaded_pipe_type = "" # "sketch2img" or "txt2img"
def load_pipe(type, weight_name):
  global pipe
  global loaded_pipe_type
  if type == "sketch2img" and loaded_pipe_type != "sketch2img":
    print("loading sketch2img pipe...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, adapter=adapter, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16").to("cuda")
    loaded_pipe_type = "sketch2img"
  elif type == "txt2img" and loaded_pipe_type != "txt2img":
    print("loading txt2img pipe...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, scheduler=euler_a, torch_dtype=torch.float16, variant="fp16").to("cuda")
    loaded_pipe_type = "txt2img"
  if weight_name and weight_name != " ":
    pipe.load_lora_weights(finetunes_url, weight_name=weight_name)
  else:
    pipe.unload_lora_weights()

def generate(sketch_img, finetune_filename, prompt, negative_prompt, steps, guidance, adapter_guidance, seed):
  if seed <= 0: seed = None
  current_seed = seed or torch.randint(0, int(1e5), size=(1, 1))[0].item()
  generator = torch.Generator().manual_seed(int(current_seed))
  if sketch_img is not None:
    load_pipe("sketch2img", finetune_filename)
    image = pidinet(sketch_img, detect_resolution=1024, image_resolution=1024, apply_filter=True)
    result = pipe(prompt=prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=steps, adapter_conditioning_scale=adapter_guidance, guidance_scale=guidance, generator=generator).images[0]
  else:
    load_pipe("txt2img", finetune_filename)
    result = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator).images[0]
  result.save(f"{results_url}/{datetime.datetime.now().strftime('%y.%m.%d_%H-%M-%S')}_{current_seed}.png")
  return [
    result,
    f"""
    Сохранено в ... {123}
    123
    asdf
    """
  ]

ui = gr.Interface(
  fn=generate,
  inputs=[
    gr.Image(value="https://i.imgur.com/m92Alz2.png", label="Скетч (sketch)", width=1024*0.71, height=1024*0.71),
    gr.Dropdown(label="Файнтюн версия (finetune lora weights)", choices=finetune_choices, value=finetune_choices[0]),
    gr.Textbox(label="Промпт (prompt)", value=default_prompt),
    gr.Textbox(label="Негативный промпт (negative prompt)", value=default_negative_prompt),
    gr.Slider(label="Шаги (steps)", minimum=0, maximum=50, step=1, value=50),
    gr.Slider(label="Строгость промпта (guidance)", minimum=0, maximum=10, step=0.5, value=8),
    gr.Slider( label="Строгость скетча (sketch adapter guidance)", minimum=0, maximum=1, step=0.1, value=0.9),
    gr.Number(label="Сид (seed)", value=-1),
  ],
  outputs=[
    gr.Image(label="Результат (generation result)"),
    gr.Markdown(value=""),
  ],
  description="Описание и инструкция будут тут ...",
  allow_flagging=False,
)

if __name__ == "__main__":
  ui.launch(show_api=False, inbrowser=True)
