import os

from torch.utils.data import Dataset
import torch

import json
from PIL import Image
from torchvision import transforms
from decord import VideoReader, cpu
import pandas as pd
import random
import torchvision.transforms.functional as TF
from tqdm import tqdm


class AspectRatioResizeCenterCrop:
    def __init__(self, target_h, target_w):
        self.target_h = target_h
        self.target_w = target_w

    def __call__(self, img):
        orig_w, orig_h = TF.get_image_size(img)

        scale = max(self.target_h / orig_h, self.target_w / orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        img_resized = TF.resize(img, [new_h, new_w], antialias=True)

        img_cropped = TF.center_crop(img_resized, [self.target_h, self.target_w])
        
        return img_cropped


class ResizeAndPad:
    def __init__(self, target_height, target_width, fill_color=(255, 255, 255)):
        self.target_height = target_height
        self.target_width = target_width
        self.fill_color = fill_color

    def __call__(self, img):
        w, h = img.size
        scale = min(self.target_width / w, self.target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_left = (self.target_width - new_w) // 2
        pad_top = (self.target_height - new_h) // 2
        pad_right = self.target_width - new_w - pad_left
        pad_bottom = self.target_height - new_h - pad_top

        padding = (pad_left, pad_top, pad_right, pad_bottom)

        return TF.pad(img, padding, fill=self.fill_color, padding_mode='constant')


class FashionVideoDataset(Dataset):
    def __init__(
        self,
        meta_paths,
        aspect_ratios={'1.78': [1280.0, 704.0],},
        num_frames=81,
        mixed_captions=False,
    ):
        self.meta_data = pd.concat(
            [pd.read_csv(p)[['video', 'prompt', 'width', 'height']] for p in meta_paths],
            ignore_index=True
        )

        self.aspect_ratios = aspect_ratios
        self.num_frames = num_frames
        self.mixed_captions = mixed_captions

        self.transforms = {
            str(ratio): transforms.Compose([
                    AspectRatioResizeCenterCrop(hw[0], hw[1]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                ])
                for ratio, hw in aspect_ratios.items()
            }

        self.transforms2 = {
            str(ratio): transforms.Compose([
                ResizeAndPad(hw[0], hw[1], fill_color=(255, 255, 255)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
            for ratio, hw in aspect_ratios.items()
        }

        # 初始化时分桶
        self.bucket_indexs = self._build_bucket_indexs()

    def _build_bucket_indexs(self):
        bucket_indexs = {str(ratio): [] for ratio in self.aspect_ratios}
        for i, row in tqdm(self.meta_data.iterrows(), total=len(self.meta_data), desc="Building ratio index"):
            w, h = row.get("width"), row.get("height")
            if pd.isna(w) or pd.isna(h):
                continue
            ratio   = int(h) / int(w)
            closest = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            bucket_indexs[str(closest)].append(i)

        return bucket_indexs

    def _get_closest_ratio(self, idx):
        row = self.meta_data.iloc[idx]
        w, h = row.get("width"), row.get("height")
        ratio   = int(h) / int(w)
        closest = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))

        return row, str(closest)

    def __len__(self):
        return sum(len(v) for v in self.bucket_indexs.values())
    
    def complete_fields(self, data):
        d = data.copy() if isinstance(data, dict) else {}

        cn_short = d.get("cn_short", None)
        en_short = d.get("en_short", None)
        cn_long = d.get("cn_long", None)
        en_long = d.get("en_long", None)

        if cn_short is None and en_short is not None:
            cn_short = en_short
        elif en_short is None and cn_short is not None:
            en_short = cn_short

        if cn_long is None and en_long is not None:
            cn_long = en_long
        elif en_long is None and cn_long is not None:
            en_long = cn_long

        if cn_short is None and en_short is None and cn_long is not None and en_long is not None:
            cn_short = cn_long
            en_short = en_long

        if cn_long is None and en_long is None and cn_short is not None and en_short is not None:
            cn_long = cn_short
            en_long = en_short

        d["cn_short"] = cn_short
        d["en_short"] = en_short
        d["cn_long"] = cn_long
        d["en_long"] = en_long

        return d

    def __getitem__(self, idx):
        while True:
            try:
                row_data, closest_ratio = self._get_closest_ratio(idx)

                prompts = json.loads(row_data['prompt'])
                prompts = self.complete_fields(prompts)

                if not self.mixed_captions: # post-training
                    prompt = prompts["en_long"][0] + " " + prompts["en_long"][1]
                else: # pre-training 
                    if random.random() < 0.7:
                        prompt = prompts["en_long"][0]
                    else:
                        prompt = prompts["en_long"][0] + " " + prompts["en_long"][1]

                video_path = row_data['video']
                src_path = video_path.replace(".mp4", "_src.png")
                cloth_path = video_path.replace(".mp4", "_cloth.png")
                
                vr = VideoReader(video_path, ctx=cpu(0))
                video = vr.get_batch(range(self.num_frames)).asnumpy()

                video = [Image.fromarray(frame).convert('RGB') for frame in video]
                video = torch.stack([self.transforms[closest_ratio](frame) for frame in video], dim=0)

                src_image = self.transforms[closest_ratio](Image.open(src_path).convert('RGB'))
                cloth_image = self.transforms2[closest_ratio](Image.open(cloth_path).convert('RGB'))

                return {
                    'video': video,  # (T, C, H, W)
                    'cloth_image': cloth_image,
                    'src_image': src_image,
                    'prompt': prompt,
                }

            except Exception as e:
                print('Error occurred while processing video:', video_path)
                print(f'Detailed error: {e}')
                closest_ratio = self._get_closest_ratio(idx)
                same_bucket_index = self.bucket_indexs[closest_ratio]
                idx = random.choice(same_bucket_index)
                continue
    
    def collate_fn(self, batchs):
        video = torch.stack([example["video"] for example in batchs])
        cloth_image = torch.stack([example["cloth_image"] for example in batchs])
        src_image = torch.stack([example["src_image"] for example in batchs])
        prompt = [example['prompt'] for example in batchs]
        batch_dict = {
            "video": video,
            "cloth_image": cloth_image,
            "src_image": src_image,
            "prompt": prompt,
        }
        return batch_dict
