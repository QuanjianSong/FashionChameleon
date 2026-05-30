from backbones.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from typing import List
import torch
from tqdm import tqdm
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import torch.nn.functional as F
import peft


class Wan22ICInferencePipeline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.timestep_shift = getattr(args, "timestep_shift", 5.0)

        # initialize all models
        self.generator_model_name = getattr(args, "generator_name", args.model_name)
        self.generator = WanDiffusionWrapper(
            **getattr(self.args, "model_kwargs", {}),
            model_name=self.args.model_name,
            seq_list=list(self.args.image_or_video_shape),
            is_causal=False,
        )
        self.generator.requires_grad_(False)

        if self.args.use_lora:
            print("Applying LoRA to models...")
            self.generator.model = self._configure_lora_for_model(self.generator.model, "teacher")
            self.generator.model.requires_grad_(False)

        self.text_encoder = WanTextEncoder(model_name=self.args.model_name)
        self.text_encoder.requires_grad_(False)
        self.vae = WanVAEWrapper(model_name=self.args.model_name)
        self.vae.requires_grad_(False)

    def _configure_lora_for_model(self, transformer, model_name):
        """Configure LoRA for a WanDiffusionWrapper model"""
        # Find all Linear modules in WanAttentionBlock modules
        target_linear_modules = set()
        
        # Define the specific modules we want to apply LoRA to
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

        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")

        # create LoRA config
        peft_config = peft.LoraConfig(
            r=self.args.rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=target_linear_modules,
        )
        # apply LoRA to the transformer
        lora_model = peft.get_peft_model(transformer, peft_config)

        print('peft_config', peft_config)
        lora_model.print_trainable_parameters()

        return lora_model

    @torch.no_grad()
    def inference(self, noise: torch.Tensor, text_prompts: List[str], src_data, cloth_data):
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )
        latents = noise

        sample_scheduler = self._initialize_sample_scheduler(noise)
        
        for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
            latent_model_input = torch.cat([
                src_data,
                cloth_data,
                latents,
            ], dim=1)            

            timestep = t * torch.ones([latents.shape[0], latents.shape[1]], device=noise.device, dtype=noise.dtype)
            timestep = F.pad(timestep, pad=(latent_model_input.shape[1] - latents.shape[1], 0), mode="constant", value=0)
            
            if '2.2' in self.args.model_name and '5B' in self.args.model_name:
                temp_ts = timestep[:, :, None, None].expand(-1, -1, latents.shape[-2] // 2, latents.shape[-1] // 2)
                temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
                wan22_input_timestep = temp_ts.to(noise.device, dtype=torch.long)
            else:
                wan22_input_timestep = None
            conditional_dict['wan22_input_timestep'] = wan22_input_timestep
            unconditional_dict['wan22_input_timestep'] = wan22_input_timestep

            with (
                    torch.amp.autocast('cuda', dtype=torch.bfloat16),
                    torch.no_grad(),
            ):
                flow_pred_cond, _ = self.generator(latent_model_input, conditional_dict, timestep)
                flow_pred_uncond, _ = self.generator(latent_model_input, unconditional_dict, timestep)

            flow_pred = flow_pred_uncond[:, -latents.shape[1]:, :, :, :] + self.args.guidance_scale * (
                flow_pred_cond[:, -latents.shape[1]:, :, :, :] - flow_pred_uncond[:, -latents.shape[1]:, :, :, :])

            temp_x0 = sample_scheduler.step(
                flow_pred.unsqueeze(0),
                t,
                latents.unsqueeze(0),
                return_dict=False)[0]
            latents = temp_x0.squeeze(0)

        x0 = latents
        video = self.vae.decode_to_pixel(x0)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        del sample_scheduler

        return video
    
    def _initialize_sample_scheduler(self, noise):
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            50, device=noise.device, shift=self.timestep_shift)
        self.timesteps = sample_scheduler.timesteps

        return sample_scheduler
