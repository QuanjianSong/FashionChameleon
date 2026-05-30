<div align="center">
<h1>
<img src="assets/fashionchameleon.png" width="40" style="vertical-align: middle;" />
FashionChameleon: Towards Real-Time and Interactive Human-Garment Video Customization
</h1>

<p align="center">
    <span>
        <a href="https://arxiv.org/pdf/2605.15824" target="_blank"> 
        <img src='https://img.shields.io/badge/2605.15824-FashionChameleon-red' alt='Paper PDF'></a> &emsp;  &emsp; 
    </span>
    <span> 
        <a href='https://quanjiansong.github.io/projects/FashionChameleon' target="_blank">
        <img src='https://img.shields.io/badge/Project_Page-FashionChameleon-green' alt='Project Page'></a>  &emsp;  &emsp;
    </span>
    <br>
    <span> 
        <a href='https://huggingface.co/papers/2605.15824' target="_blank"> 
        <img src='https://img.shields.io/badge/Hugging_Face-FashionChameleon-blue' alt='Hugging Face'></a> &emsp;  &emsp;
    </span>
    <span> 
        <a href='https://huggingface.co/datasets/QuanjianSong/HGC-Bench' target="_blank"> 
        <img src='https://img.shields.io/badge/Hugging_Face-HGC--Bench-yellow' alt='HGC-Bench'></a> &emsp;  &emsp;
    </span>
</p>

<br/>

<div align="center">
<b>TL;DR:</b><br/>
We propose <span className="text-white font-medium">FashionChameleon</span>, a real-time and interactive framework for human-garment customization in streaming autoregressive video generation.  
It achieves real-time generation at 23.8 FPS on a single GPU.
</div>
<img src="assets/teaser.png" style="width:100%; height:100%;"/>


</div>


## 📅 Todo
- [ ] Release the checkpoint.
- [ ] Release the training-free kv cache rescheduling for interactive inference.
- [x] 🔥 Release the code (Wan2.2-TI2V-5B) for gradient-reweighted dmd and the corresponding inference.
- [x] 🔥 Release the code (Wan2.2-TI2V-5B) for in-context teacher forcing and the corresponding inference.
- [x] 🔥 Release the code (Wan2.2-TI2V-5B) for in-context sft and the corresponding inference.
- [x] 🔥 Release the <a href="https://huggingface.co/datasets/QuanjianSong/HGC-Bench" target="_blank">HGC-Bench</a>.
- [x] 🔥 Release the <a href="https://quanjiansong.github.io/projects/FashionChameleon" target="_blank">Project Page</a>.
- [x] 🔥 Release the <a href="https://arxiv.org/pdf/2605.15824" target="_blank">Technical Report</a>.


## ✨ Highlight
> **1. Interactive Customization.** We train a single-garment switching teacher using tailored I2V priors and mismatched reference–garment pairs. During generation, we introduce KV-cache rescheduling to enable interactive multi-garment customization without requiring video data containing multi-garment switching.

> **2. Gradient-Reweighted DMD.** Traditional self-forcing treats all self-rolled frames equally during DMD backpropagation. However, later frames typically suffer from larger quality degradation and thus require stronger gradient supervision. We dynamically reweight DMD gradients during self-rolling using a reward model to improve extrapolation consistency.

> **3. Real-Time Generation.** Through streaming distillation with in-context learning, FashionChameleon achieves 23.8 FPS for 720p generation on a single H200 GPU, 30–180× faster than existing customization methods.

<img src="assets/intro.png" style="width:100%; height:100%;"/>


## 🎬 Overview
***FashionChameleon*** is built upon [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), featuring: **(i)** Teacher Model with In-Context Learning, **(ii)** Streaming Distillation with In-Context Learning, and **(iii)** Training-Free KV Cache Rescheduling.
<img src="assets/overall_framework.png" style="width:100%; height:100%;"/>


## 🔧 Step0. Setup
### Prepare Environment
```
git clone https://github.com/QuanjianSong/FashionChameleon.git
cd FashionChameleon
# Installation with the requirement.txt
conda create -n FashionChameleon python=3.10
conda activate FashionChameleon
pip install -r requirements.txt
```

### Download Backbone
Our FashionChameleon is built upon [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B). You should first download the backbone weight by running:
```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir-use-symlinks False --local-dir checkpoints/wan_models/Wan2.2-TI2V-5B
```

In addition, you need to download the [CLIP-ViT](https://huggingface.co/openai/clip-vit-large-patch14) weights for aesthetic reward initialization by running:
```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openai/clip-vit-large-patch14 --local-dir-use-symlinks False --local-dir checkpoints/clip-vit-large-patch14
```

As well as download the [PredictionHead](https://github.com/christophschuhmann) for aesthetic reward by running:
```
wget -c https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth -O checkpoints/sac+logos+ava1-l14-linearMSE.pth
```


## 🚀 Step1. In-Context SFT  
### Start Training
You can run the following command to start sft:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=8989 train.py \
    --config_path configs/sft_wan22_ic.yaml \
    --save_dir outputs/sft_wan22_ic
```
or simply run:
```bash
bash scripts/train/sft_wan22_ic.sh
```
All training configurations are recorded in `configs/sft_wan22_ic.yaml`, which can be freely modified according to your needs.
Note that our training framework supports **variable-resolution bucketing strategies**, **gradient accumulation**, and **mixed captions**; you only need to adjust the corresponding `ASPECT_RATIO`, `grad_accum_steps`, and `mixed_captions` parameters accordingly.
Our FashionChameleon keeps a fixed training resolution of 1280 × 704 while simultaneously maintaining mixed captions during the in-context sft process, with a global batch size of 64.

### Start Inference
You can run the following command to start bidirectional inference:
```
CUDA_VISIBLE_DEVICES=1 python predictor/infer_ic.py --config_path configs/sft_wan22_ic.yaml \
    --seed 42 \
    --h 1280 \
    --w 704 \
    --num_frames 81 \
    --output_path samples/sft_wan22_ic/ \
    --checkpoint XXX
```
or simple run:
```bash
bash scripts/infer/infer_wan22_ic.sh
```
The `checkpoint` denotes the weights after in-context sft.
Our inference code by default processes data in the format of HGC-Bench. You can first download the test dataset from [Hugging Face](https://huggingface.co/datasets/QuanjianSong/HGC-Bench).


## 🧑‍🏫 Step2. In-Context Teacher Forcing
### Start Training
You can run the following command to start in-context teacher forcing:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=7777 train.py \
    --config_path configs/tf_wan22_ic.yaml \
    --save_dir outputs/tf_wan22_ic
```
or simple run:
```bash
bash scripts/train/tf_wan22_ic.sh
```
All training configurations are recorded in `configs/tf_wan22_ic.yaml`, which can be freely modified according to your needs.
Note that our training framework supports **variable-resolution bucketing strategies**, **gradient accumulation**, and **mixed captions**; you only need to adjust the corresponding `ASPECT_RATIO`, `grad_accum_steps`, and `mixed_captions` parameters accordingly.
Our FashionChameleon keeps a fixed training resolution of 1280 × 704 while simultaneously maintaining long caption during the in-context teacher forcing, with a global batch size of 64.

### Start Inference
You can run the following command to start causal inference:
```
CUDA_VISIBLE_DEVICES=1 python predictor/causal_infer_ic.py --config_path configs/tf_wan22_ic.yaml \
    --seed 42 \
    --h 1280 \
    --w 704 \
    --num_frames 81 \
    --output_path samples/tf_wan22_ic/ \
    --checkpoint XXX
```
or simple run:
```bash
bash scripts/infer/causal_infer_wan22_ic.sh
```
The `checkpoint` denotes the weights after in-context teacher forcing.
Our inference code by default processes data in the format of HGC-Bench. You can first download the test dataset from [Hugging Face](https://huggingface.co/datasets/QuanjianSong/HGC-Bench).


## 📈 Step3. Gradient-Reweighted DMD
### Start Training
You can run the following command to start gradient-reweighted dmd with self-forcing:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=7777 train.py \
    --config_path configs/gr_dmd_wan22_ic.yaml \
    --save_dir outputs/gr_dmd_wan22_ic
```
or simple run:
```bash
bash scripts/train/sf_wan22_ic.sh
```
All training configurations are recorded in `configs/gr_dmd_wan22_ic.yaml`, which can be freely modified according to your needs.
Note that our training framework supports **variable-resolution bucketing strategies**, **gradient accumulation**, and **mixed captions**; you only need to adjust the corresponding `ASPECT_RATIO`, `grad_accum_steps`, and `mixed_captions` parameters accordingly.
Our FashionChameleon keeps a fixed training resolution of 1280 × 704 while simultaneously maintaining long caption during the in-context gradient-reweighted dmd, with a global batch size of 64..
### Start Inference
You can run the following command to start stream inference:
```
CUDA_VISIBLE_DEVICES=1 python predictor/stream_infer_ic.py --config_path configs/gr_dmd_wan22_ic.yaml \
    --seed 42 \
    --h 1280 \
    --w 704 \
    --num_frames 81 \
    --output_path samples/gr_dmd_wan22_ic_reward/ \
    --checkpoint XXX
```
or simple run:
```bash
bash scripts/infer/stream_infer_wan22_ic.sh
```
The `checkpoint` denotes the weights after gradient-reweighted dmd with self-forcing.
Our inference code by default processes data in the format of HGC-Bench. You can first download the test dataset from [Hugging Face](https://huggingface.co/datasets/QuanjianSong/HGC-Bench).


<!-- ## 🔧 KV cache rescheduling for interactive customization -->


## 🌈 Comparison
<img src="assets/comparison.png" style="width:100%; height:100%;"/>



## 🌊 Application
<img src="assets/application.png" style="width:100%; height:100%;"/>


## 🤝 Acknowledgements
This codebase borrows from [Wan2.2](https://github.com/Wan-Video/Wan2.2) and [Self-Forcing](https://github.com/guandeh17/self-forcing). Many thanks to them for open-source contributions.


## 🎓 Bibtex
🤗 If you find this code helpful for your research, please cite:
```
@article{song2026fashionchameleon,
  title={FashionChameleon: Towards Real-Time and Interactive Human-Garment Video Customization},
  author={Song, Quanjian and Shen, Yefeng and Chen, Mengting and Sun, Hao and Lan, Jinsong and Zhu, Xiaoyong and Zheng, Bo and Cao, Liujuan},
  journal={arXiv preprint arXiv:2605.15824},
  year={2026}
}
```
