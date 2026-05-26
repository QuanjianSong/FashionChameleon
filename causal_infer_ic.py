from pipelines import CausalWan22ICInferencePipeline
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
import argparse
import torch
import os
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import pandas as pd
import json


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output_path", type=str)
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--h", type=int, default=480)
parser.add_argument("--w", type=int, default=832)
parser.add_argument("--num_frames", type=int, default=81)
args = parser.parse_args()
assert args.num_frames % 4 == 1, "num_frames must be 1 more than a multiple of 4"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

os.makedirs(args.output_path, exist_ok=True)
config = OmegaConf.load(args.config_path)
config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"),
        OmegaConf.load(args.config_path),
        OmegaConf.create(vars(args))
    )

config.local_attn_size = list(config.image_or_video_shape)[1]

pipe = CausalWan22ICInferencePipeline(config)
if args.checkpoint is not None:
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if 'generator_ema' in state_dict:
        state_dict = state_dict['generator_ema']
    elif 'generator' in state_dict:
        state_dict = state_dict['generator']
    pipe.generator.load_state_dict(state_dict)
pipe = pipe.to(device="cuda", dtype=torch.bfloat16)

root_path = "./HGC-Bench/"
meta_data = pd.read_csv(os.path.join(root_path, 'meta_data.csv'))

for idx in range(len(meta_data)):
    try:
        row_data = meta_data.iloc[idx]

        prompt = json.loads(row_data['prompt'])['caption']
        prompt = prompt['en_short']

        src_path = row_data['src_image']
        cloth_path = row_data['cloth_image']

        save_path = os.path.join(args.output_path, f'output_{idx}.mp4')
        if os.path.exists(save_path):
            print(f"{save_path} exists, skip")
            continue

        src_data = Image.open(src_path).convert("RGB")
        src_data = ImageOps.fit(src_data, (args.w, args.h), method=Image.LANCZOS, centering=(0.5, 0.5)) # keep ratio and crop
        src_data = TF.to_tensor(src_data).sub_(0.5).div_(0.5).to("cuda").unsqueeze(1).to(dtype=torch.bfloat16)
        src_data = pipe.vae.encode_to_latent(src_data.unsqueeze(0)).to(dtype=torch.bfloat16)

        cloth_data = Image.open(cloth_path).convert("RGB")
        cloth_data = ImageOps.pad(cloth_data, (args.w, args.h), color=(255, 255, 255), centering=(0.5, 0.5)) # keep ratio and padding
        cloth_data = TF.to_tensor(cloth_data).sub_(0.5).div_(0.5).to("cuda").unsqueeze(1).to(dtype=torch.bfloat16)
        cloth_data = pipe.vae.encode_to_latent(cloth_data.unsqueeze(0)).to(dtype=torch.bfloat16)

        video = (
            pipe.causal_inference(
                noisy_image_or_video=torch.randn(
                    1,
                    (args.num_frames - 1) // 4 + 1,
                    48,
                    args.h // 16,
                    args.w // 16,
                    generator=torch.Generator(device="cuda").manual_seed(args.seed),
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                text_prompts=[prompt],
                src_data=src_data,
                cloth_data=cloth_data,
            )[0]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        export_to_video(video, save_path, fps=24)

    except Exception as e:
        print(f"[idx={idx}] error: {e}")
        continue
