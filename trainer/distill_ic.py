import os

import gc
from tqdm import tqdm

from datasets import cycle, FashionVideoDataset, BucketSampler
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.util import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from models import GradientReweightedDMDModel
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
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
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
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
        self.model = GradientReweightedDMDModel(config, device=self.device)
        ##############################################################################################################

        # (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(self.config, "generator_ckpt", False):
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            #
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )
            if self.global_rank == 0:
                print(f"Loading pretrained generator from {self.config.generator_ckpt}")
        if getattr(self.config, "teacher_ckpt", False):
            state_dict = torch.load(config.teacher_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            #
            self.model.real_score.load_state_dict(
                state_dict, strict=True
            )
            self.model.fake_score.load_state_dict(
                state_dict, strict=True
            )
            if self.global_rank == 0:
                print(f"Loading pretrained real_score and fake_score from {self.config.teacher_ckpt}")

        # load lora
        if getattr(config, "use_lora", False):
            if self.global_rank == 0:
                print("Applying LoRA to models...")
            self.model.generator.model = self._configure_lora_for_model(self.model.generator.model, "generator")
            self.model.fake_score.model = self._configure_lora_for_model(self.model.generator.model, "fake_score")

        ##############################################################################################################
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )
        #
        self.model.vae = self.model.vae.to(device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
        if getattr(self.config, "use_gradient_reweighted", False):
            self.model.reward_model = self.model.reward_model.to(device=self.device)
        ##############################################################################################################
        # Set up EMA parameter containers
        if self.step >= self.config.ema_start_step and (self.config.ema_weight is not None) and (self.config.ema_weight > 0.0):
            print(f"Setting up EMA with weight {self.config.ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
        else:
            self.generator_ema = None
        ##############################################################################################################
        # configure optimizers
        self.generator_optimizer, self.critic_optimizer = self.configure_optimizers() 
        # configure dataloader
        self.dataloader = self.configure_dataloader()
        ##############################################################################################################

    def _configure_lora_for_model(self, transformer, model_name):
        """Configure LoRA for a WanDiffusionWrapper model"""
        # find all Linear modules in WanAttentionBlock modules
        target_linear_modules = set()
        
        # define the specific modules we want to apply LoRA to
        if model_name == 'teacher':
            adapter_target_modules = ['WanAttentionBlock']
        elif model_name == 'generator':
            adapter_target_modules = ['CausalWanAttentionBlock']
        elif model_name == 'fake_score':
            adapter_target_modules = ['WanAttentionBlock']
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        for name, module in transformer.named_modules():
            if module.__class__.__name__ in adapter_target_modules:
                for full_submodule_name, submodule in module.named_modules(prefix=name):
                    if isinstance(submodule, torch.nn.Linear):
                        target_linear_modules.add(full_submodule_name)
        
        target_linear_modules = list(target_linear_modules)
        
        if self.global_rank == 0:
            print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        
        # create LoRA config
        peft_config = peft.LoraConfig(
            r=self.args.rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=target_linear_modules,
        )

        # apply LoRA to the transformer
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
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )
        critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=self.config.lr_critic if hasattr(self.config, "lr_critic") else self.config.lr,
            betas=(self.config.beta1_critic, self.config.beta2_critic),
            weight_decay=self.config.weight_decay
        )

        return generator_optimizer, critic_optimizer

    def configure_dataloader(self):
        # dataset
        dataset = FashionVideoDataset(
            meta_paths=list(self.config.meta_paths),
            aspect_ratios=self.config.ASPECT_RATIO,
            num_frames=81,
            mixed_captions=self.config.mixed_captions, # long caption
        )
        # batch_sampler
        batch_sampler = BucketSampler(
            bucket_indexs=dataset.bucket_indexs,
            aspect_ratios=dataset.aspect_ratios,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.config.seed,
        )
        # dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler, 
            collate_fn=dataset.collate_fn,
            num_workers=8,
        )
        # waiting
        if dist.is_initialized():
            dist.barrier()
        # log
        if self.global_rank == 0:
            print("DATASET SIZE %d" % len(dataset))

        return cycle(dataloader)

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
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

    def fwdbwd_one_step(self, train_generator):
        self.model.eval() # prevent any randomness (e.g. dropout)
        mean_loss = 0

        for _ in range(self.config.grad_accum_steps):
            batch = next(self.dataloader)

            text_prompts = batch["prompt"]
            src_data = batch["src_image"].to(device=self.device, dtype=self.dtype)
            cloth_data = batch["cloth_image"].to(device=self.device, dtype=self.dtype)
            
            batch_size = len(text_prompts)
            image_or_video_shape = list(self.config.image_or_video_shape)
            image_or_video_shape[0] = batch_size

            # extract the conditional infos
            with torch.no_grad():
                src_data = self.model.vae.encode_to_latent(src_data.unsqueeze(2))
                cloth_data = self.model.vae.encode_to_latent(cloth_data.unsqueeze(2))
                
                conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
                if not getattr(self, "unconditional_dict", None):
                    unconditional_dict = self.model.text_encoder(
                        text_prompts=[self.config.negative_prompt] * batch_size)
                    unconditional_dict = {k: v.detach()
                                        for k, v in unconditional_dict.items()}
                    self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
                else:
                    unconditional_dict = self.unconditional_dict
            # append condition
            conditional_dict['src_data'] = src_data
            unconditional_dict['src_data'] = src_data
            conditional_dict['cloth_data'] = cloth_data
            unconditional_dict['cloth_data'] = cloth_data

            # store gradients for the generator (if training the generator)
            if train_generator:
                generator_loss, generator_log_dict = self.model.generator_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    use_gradient_reweighted=getattr(self.config, "use_gradient_reweighted", False),
                )
                generator_loss = generator_loss / self.config.grad_accum_steps
                generator_loss.backward()
                mean_loss += generator_loss.detach()
            else:
                # store gradients for the critic (if training the critic)
                critic_loss, critic_log_dict = self.model.critic_loss(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                )
                critic_loss = critic_loss / self.config.grad_accum_steps
                critic_loss.backward()
                mean_loss += critic_loss.detach()

        if train_generator:
            grad_norm = self.model.generator.clip_grad_norm_(
                self.config.max_grad_norm_generator
            )
            return {
                "generator_loss": mean_loss,
                "generator_grad_norm": grad_norm,
            }
        else:
            grad_norm = self.model.fake_score.clip_grad_norm_(
                self.config.max_grad_norm_critic
            )
            return {
                "critic_loss": mean_loss,
                "critic_grad_norm": grad_norm,
            }

    def train(self):
        start_step = self.step
        self.progress_bar = tqdm(range(self.step, self.config.max_step), initial=self.step, desc="distillation")

        while True:
            if self.step % self.config.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                generator_log_dict = self.fwdbwd_one_step(True)
                self.generator_optimizer.step()

                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
            # train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_log_dict = self.fwdbwd_one_step(False)
            self.critic_optimizer.step()            

            # ---------------------------------------------------------------
            self.step += 1
            # ---------------------------------------------------------------

            # create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # save the model
            if (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # logging
            if self.global_rank == 0:
                log_dict = {}
                if TRAIN_GENERATOR:
                    if getattr(self.config, "use_gradient_reweighted", False):
                        log_dict.update(
                            {
                                "generator_loss": generator_log_dict["generator_loss"].item(),
                                "generator_grad_norm": generator_log_dict["generator_grad_norm"].item(),
                            }
                        )
                    else:
                        log_dict.update(
                            {
                                "generator_loss": generator_log_dict["generator_loss"].item(),
                                "generator_grad_norm": generator_log_dict["generator_grad_norm"].item(),
                            }
                        )

                log_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].item()
                    }
                )
                # log in the terminal
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({
                    "step": self.step, 
                    "generator_loss": f"{log_dict['generator_loss']:.6f}" if TRAIN_GENERATOR else 0,
                    "critic_loss": f"{log_dict['critic_loss']:.6f}",
                })
                # log in wandb or tensorboard
                self.log_metrics(log_dict)

            if self.step % self.config.log_iters == 0:
                gc.collect()
                torch.cuda.empty_cache()
                # ****************************************************************************************************
                self.validate()
                # ****************************************************************************************************
                gc.collect()
                torch.cuda.empty_cache()

            if self.step >= self.config.max_step:
                break

    @torch.no_grad()
    def validate(self):
        batch = next(self.dataloader)

        text_prompts = batch["prompt"]
        src_data = batch["src_image"].to(
            device=self.device, dtype=self.dtype
        )
        cloth_data = batch["cloth_image"].to(
            device=self.device, dtype=self.dtype
        )

        batch_size = len(text_prompts)  
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        # remove the in-context
        image_or_video_shape[1] = image_or_video_shape[1] - 2

        # extract the conditional infos
        with torch.no_grad():
            src_data = self.model.vae.encode_to_latent(src_data.unsqueeze(2))
            cloth_data = self.model.vae.encode_to_latent(cloth_data.unsqueeze(2))
            
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
        # append addition
        conditional_dict['src_data'] = src_data
        conditional_dict['cloth_data'] = cloth_data

        noisy_image_or_video = torch.randn(image_or_video_shape, device=self.device, dtype=self.dtype)

        # inference
        if self.model.inference_pipeline is None:
            self.model._initialize_inference_pipeline()
        pred_image_or_video = self.model.inference_pipeline.validate(
            noisy_image_or_video=noisy_image_or_video, 
            **conditional_dict,
        )

        # decode
        video = self.model.vae.decode_to_pixel(pred_image_or_video, use_cache=False)
        video = 255.0 * (video * 0.5 + 0.5).clamp(0, 1).cpu().numpy()

        if self.global_rank == 0:
            self.log_videos(video, text_prompts)
        if dist.is_initialized():
            dist.barrier()

    def log_metrics(self, metrics: dict):
        if self.config.logger_type == 'wandb' and self.logger is not None:
            self.logger.log(metrics, step=self.step)
        elif self.config.logger_type == 'tensorboard' and self.logger is not None:
            for k, v in metrics.items():
                self.logger.add_scalar(k, v, self.step)
        else:
            pass

    def log_videos(self, video, text_prompts):
        batch_size = video.shape[0]
        max_nums = min(batch_size, 4)
        # wandb
        if self.config.logger_type == "wandb" and self.logger is not None:
            vids = [
                wandb.Video(
                    video[i],
                    fps=24,
                    format="mp4",
                    caption=text_prompts[i],
                ) for i in range(max_nums)
            ]
            self.logger.log({"val": vids}, step=self.step)
        # tensorBoard
        elif self.config.logger_type == "tensorboard" and self.logger is not None:
            video_tensor = torch.from_numpy(video).float() / 255.0
            for i in range(max_nums):
                self.logger.add_video(
                    f"{text_prompts[i]}",
                    video_tensor[i:i+1],
                    global_step=self.step,
                    fps=24,
                )
