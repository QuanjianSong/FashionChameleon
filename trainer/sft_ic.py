import os

import gc
from tqdm import tqdm

from datasets import cycle, FashionVideoDataset, BucketSampler
from utils.distributed import fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.util import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from models import DiffusionICModel
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
import random
import peft


class Trainer:
    def __init__(self, config):
        self.step = 0
        self.config = config
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ##############################################################################################################
        # Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.dtype = torch.float32
        self.device = torch.cuda.current_device()
        # configure logger
        self.configure_logger()
        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()
        set_seed(config.seed + self.global_rank)
        ##############################################################################################################
        self.model = DiffusionICModel(config, device=self.device)
        ##############################################################################################################

        if self.config.use_lora:
            if self.global_rank == 0:
                print("Applying LoRA to models...")
            self.model.generator.model = self._configure_lora_for_model(self.model.generator.model, "generator")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )
        self.model.vae = self.model.vae.to(device=self.device, dtype=torch.float32)
        if getattr(config, "generator_ckpt", False):
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            self.model.generator.load_state_dict(
                state_dict['generator'], strict=True
            )
            self.step = state_dict['global_step']
            if self.global_rank == 0:
                print(f"Loading pretrained generator from {self.config.generator_ckpt}, step={self.step}")
        ##############################################################################################################
        # configure optimizers
        self.generator_optimizer, self.scheduler = self.configure_optimizers() 
        # configure dataloader
        self.dataloader = self.configure_dataloader()
        ##############################################################################################################

    def _configure_lora_for_model(self, transformer, model_name):
        """Configure LoRA for a WanDiffusionWrapper model"""
        # Find all Linear modules in WanAttentionBlock modules
        target_linear_modules = set()

        # Define the specific modules we want to apply LoRA to
        adapter_target_modules = self.config.adapter_target_modules

        for name, module in transformer.named_modules():
            if module.__class__.__name__ in adapter_target_modules:
                for full_submodule_name, submodule in module.named_modules(prefix=name):
                    if isinstance(submodule, torch.nn.Linear):
                        target_linear_modules.add(full_submodule_name)
        
        target_linear_modules = list(target_linear_modules)

        if self.global_rank == 0:
            print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        
        # Create LoRA config
        peft_config = peft.LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            target_modules=target_linear_modules,
        )

        # Apply LoRA to the transformer
        lora_model = peft.get_peft_model(transformer, peft_config)

        if self.global_rank == 0:
            print('peft_config', peft_config)
            lora_model.print_trainable_parameters()

        return lora_model

    def configure_logger(self):
        if self.global_rank == 0:
            exp_name = os.path.basename(self.config.config_path).split(".")[0]
            if self.config.logger_type == 'wandb':
                flag = wandb.login(host=self.config.wandb_host, key=self.config.wandb_key)
                self.logger = wandb.init(
                    project=self.config.project,
                    name=exp_name,
                    dir="./logs",
                    config=OmegaConf.to_container(self.config, resolve=True),
                    mode="online" if flag else "offline",
                )
            elif self.config.logger_type == 'tensorboard':
                tb_logdir = os.path.join('./logs/tensorboard', exp_name)
                os.makedirs(tb_logdir, exist_ok=True)
                self.logger = SummaryWriter(log_dir=tb_logdir)
            else:
                self.logger = None
        if dist.is_initialized():
            dist.barrier()

    def configure_optimizers(self):
        generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )
        #
        def lr_lambda(current_step):
            if current_step < self.config.warmup_step:
                return float(current_step) / float(max(1, self.config.warmup_step))
            elif current_step >= self.config.warmup_step and current_step < self.config.decay_step:
                return 1.0
            else:
                return 0.5
        #
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(generator_optimizer, lr_lambda=lr_lambda)
        print("Warmup steps:", self.config.warmup_step)

        return generator_optimizer, scheduler

    def configure_dataloader(self):
        # dataset
        dataset = FashionVideoDataset(
            meta_paths=list(self.config.meta_paths),
            aspect_ratios=self.config.ASPECT_RATIO,
            num_frames=81,
            mixed_caption=True
        ) 
        batch_sampler = BucketSampler(
            bucket_indexs=dataset.bucket_indexs,
            aspect_ratios=dataset.aspect_ratios,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.config.seed,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler, 
            collate_fn=dataset.collate_fn,
            num_workers=16,
        )
        #
        if dist.is_initialized():
            dist.barrier()
        if self.global_rank == 0:
            print("DATASET SIZE %d" % len(dataset))
        return cycle(dataloader)

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        state_dict = {
            "generator": generator_state_dict,
            "global_step": self.step,
        }
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.config.save_dir,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.config.save_dir,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.config.save_dir,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))
        if dist.is_initialized():
            dist.barrier()

    def fwdbwd_one_step(self):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        mean_loss = 0

        for _ in range(self.config.grad_accum_steps):
            batch = next(self.dataloader)
            text_prompts = batch["prompt"]

            drop_ratio = 0.1
            text_prompts = [prompt if random.random() > drop_ratio else '' for prompt in text_prompts]

            video_data = batch["video"].to(
                device=self.device, dtype=self.dtype
            )
            src_data = batch["src_image"].to(
                device=self.device, dtype=self.dtype
            )
            cloth_data = batch["cloth_image"].to(
                device=self.device, dtype=self.dtype
            )

            # video_data: [batch_size, num_channels, num_frames, height, width]
            video_data = self.model.vae.encode_to_latent(video_data.permute(0, 2, 1, 3, 4))
            src_data = self.model.vae.encode_to_latent(src_data.unsqueeze(2))
            cloth_data = self.model.vae.encode_to_latent(cloth_data.unsqueeze(2))

            # extract the conditional infos
            with torch.no_grad():
                conditional_dict = self.model.text_encoder(
                    text_prompts=text_prompts)
            # append condition
            conditional_dict['src_data'] = src_data
            conditional_dict['cloth_data'] = cloth_data

            
            # store gradients for the generator (if training the generator)
            generator_loss, generator_log_dict = self.model.generator_loss(
                clean_latent=video_data,
                conditional_dict=conditional_dict,
            )

            generator_loss = generator_loss / self.config.grad_accum_steps
            generator_loss.backward()
            mean_loss += generator_loss.detach()

        grad_norm = self.model.generator.clip_grad_norm_(
            self.config.max_grad_norm_generator
        )

        return {
            "generator_loss": mean_loss,
            "generator_grad_norm": grad_norm,
            "generator_log_dict": generator_log_dict, # last mini_batch
        }

    def train(self):
        start_step = self.step
        self.progress_bar = tqdm(range(self.step, self.config.max_step), initial=self.step, desc="Training")

        while True:
            if self.step % self.config.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()
   
            self.generator_optimizer.zero_grad(set_to_none=True)
            generator_log_dict = self.fwdbwd_one_step()
            self.generator_optimizer.step()
            # additional
            if self.scheduler is not None:
                self.scheduler.step()

            # ---------------------------------------------------------------
            self.step += 1
            # ---------------------------------------------------------------

            # Save the model
            if (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.global_rank == 0:
                log_dict = {}

                log_dict.update(
                    {
                        "generator_loss": generator_log_dict["generator_loss"].item(),
                        "generator_grad_norm": generator_log_dict["generator_grad_norm"].item(),
                        "generator_lr": self.generator_optimizer.param_groups[0]["lr"],
                    }
                )

                # log in the terminal
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({
                    "step": self.step, 
                    "generator_loss": f"{log_dict['generator_loss']:.6f}"
                })
                # log in wandb or tensorboard
                self.log_metrics(log_dict)

            if self.step % self.config.log_iters == 0:
                gc.collect()
                torch.cuda.empty_cache()
                # ------------------------------------------------------------
                output = generator_log_dict["generator_log_dict"]["x0_pred"]
                ground_truth = generator_log_dict["generator_log_dict"]["x0"]

                output_video = self.model.vae.decode_to_pixel(output.to(dtype=self.dtype))
                output_video = 255.0 * (output_video.cpu().numpy() * 0.5 + 0.5)

                ground_truth_video = self.model.vae.decode_to_pixel(ground_truth.to(dtype=self.dtype))
                ground_truth_video = 255.0 * (ground_truth_video.cpu().numpy() * 0.5 + 0.5)

                if self.global_rank == 0:
                    self.logger.log({"video": wandb.Video(output_video, caption="Output", fps=16, format="mp4"), "video_gt": wandb.Video(ground_truth_video, caption="Ground Truth", fps=16, format="mp4")})
                if dist.is_initialized():
                    dist.barrier()

                gc.collect()
                torch.cuda.empty_cache()

            if self.step >= self.config.max_step:
                break

    def log_metrics(self, metrics: dict):
        if self.config.logger_type == 'wandb' and self.logger is not None:
            self.logger.log(metrics, step=self.step)
        elif self.config.logger_type == 'tensorboard' and self.logger is not None:
            for k, v in metrics.items():
                self.logger.add_scalar(k, v, self.step)
        else:
            pass
