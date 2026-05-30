from backbones.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
import torch
import peft


class StreamWan22ICInferencePipeline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        # initialize all models
        self.generator_model_name = getattr(args, 'model_name')
        self.generator = WanDiffusionWrapper(
                            **getattr(self.args, "model_kwargs", {}),
                            model_name=self.generator_model_name,
                            is_causal=True,
                            seq_list=list(self.args.image_or_video_shape),
                            local_attn_size=self.args.local_attn_size,
                            sink_size=5,
                        )
        if getattr(self.args, "use_lora", False):
            print("Applying LoRA to models...")
            self.generator.model = self._configure_lora_for_model(self.generator.model, "generator")
        self.generator.requires_grad_(False)

        self.text_encoder = WanTextEncoder(model_name=self.generator_model_name)
        self.text_encoder.requires_grad_(False)
        self.vae = WanVAEWrapper(model_name=self.generator_model_name)
        self.vae.requires_grad_(False)

        # initialize all bidirectional wan hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
        
        # wan specific hyperparameters
        if "2.1" in self.generator.model_name and "1.3B" in self.generator.model_name:
            self.num_transformer_blocks = 30
            self.frame_seq_length = 1560
            self.num_frame_per_block = getattr(args, 'num_frame_per_block')
            self.context_noise = 0
            self.i2v = False
        elif "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
            self.num_transformer_blocks = 30
            self.frame_seq_length = 880
            self.num_frame_per_block = getattr(args, 'num_frame_per_block')
            self.context_noise = 0
            self.i2v = False
        else:
            raise NotImplementedError

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = getattr(args, 'independent_first_frame')
        self.num_max_frames = getattr(args, 'num_training_frames')
        self.kv_cache_size = self.num_max_frames * self.frame_seq_length

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
        
        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        
        # create LoRA config
        peft_config = peft.LoraConfig(
            r=self.args.rank,
            lora_alpha=self.args.lora_alpha,
            target_modules=target_linear_modules,        # Remove this; not needed for diffusion models
        )

        # apply LoRA to the transformer
        lora_model = peft.get_peft_model(transformer, peft_config)
  
        print('peft_config', peft_config)
        lora_model.print_trainable_parameters()

        return lora_model

    @torch.no_grad()
    def stream_inference(
        self,
        noisy_image_or_video,
        text_prompts,
        src_data=None,
        cloth_data=None,
    ):
        batch_size, num_frames, num_channels, height, width = noisy_image_or_video.shape

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts)

        if src_data is not None and cloth_data is not None:
            initial_latent = torch.cat([src_data, cloth_data], dim=1)
        else:
            initial_latent = None

        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            raise NotImplementedError
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        # initialize
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noisy_image_or_video.device,
            dtype=noisy_image_or_video.dtype
        )
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noisy_image_or_video.dtype,
            device=noisy_image_or_video.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noisy_image_or_video.dtype,
            device=noisy_image_or_video.device
        )

        # cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            # assume num_input_frames is num_input_frames + self.num_frame_per_block * num_input_blocks
            output[:, :num_input_frames] = initial_latent
            timestep = torch.ones([batch_size, num_input_frames], device=noisy_image_or_video.device, dtype=torch.int64) * 0
            if "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
                temp_ts = timestep[:, :, None, None].expand(-1, -1, height // 2, width // 2).to(device=initial_latent.device, dtype=initial_latent.dtype)
                temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
                wan22_input_timestep = temp_ts.to(initial_latent.device, dtype=torch.long)
            else:
                wan22_input_timestep = None
            # append condition
            conditional_dict['wan22_input_timestep'] = wan22_input_timestep
            self.generator(
                noisy_image_or_video=initial_latent,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length
            )
            current_start_frame += num_input_frames

        # temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for idx, current_num_frames in enumerate(all_num_frames):
            noisy_input = noisy_image_or_video[
                :, current_start_frame - num_input_frames: current_start_frame + current_num_frames - num_input_frames]

            # spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noisy_image_or_video.device,
                    dtype=torch.int64) * current_timestep

                if "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
                    temp_ts = timestep[:, :, None, None].expand(-1, -1, height // 2, width // 2).to(device=noisy_input.device, dtype=noisy_input.dtype)
                    temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
                    wan22_input_timestep = temp_ts.to(noisy_image_or_video.device, dtype=torch.long)
                else:
                    wan22_input_timestep = None
                # append condition
                conditional_dict['wan22_input_timestep'] = wan22_input_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                            [batch_size, current_num_frames], device=noisy_image_or_video.device, dtype=torch.long)
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep.flatten(0, 1),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # record the model's output
            output[:, current_start_frame: current_start_frame + current_num_frames] = denoised_pred

            # rerun with timestep zero to update KV cache using clean context
            context_timestep = self.context_noise * torch.ones(
                    [batch_size, current_num_frames], device=noisy_image_or_video.device, dtype=torch.long)

            if "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
                temp_ts = context_timestep[:, :, None, None].expand(-1, -1, noisy_input.shape[-2] // 2, noisy_input.shape[-1] // 2)
                temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
                wan22_input_timestep = temp_ts.to(noisy_image_or_video.device, dtype=torch.long)
            else:
                wan22_input_timestep = None
            # append condition
            conditional_dict['wan22_input_timestep'] = wan22_input_timestep

            # pre-filling to get kv 
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # update the start and end frame indices
            current_start_frame += current_num_frames

        # remove in-context frames
        if src_data is not None and cloth_data is not None:
            output = output[:, num_input_frames:]

        video = self.vae.decode_to_pixel(output)

        video = (video * 0.5 + 0.5).clamp(0, 1)

        return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        if "2.1" in self.generator.model_name and "1.3B" in self.generator.model_name:
            for _ in range(self.num_transformer_blocks):
                kv_cache1.append({
                    "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })
        elif "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
            for _ in range(self.num_transformer_blocks):
                kv_cache1.append({
                    "k": torch.zeros([batch_size, self.kv_cache_size, 24, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, self.kv_cache_size, 24, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })
        else:
            raise NotImplementedError

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        if "2.1" in self.generator.model_name and "1.3B" in self.generator.model_name:
            for _ in range(self.num_transformer_blocks):
                crossattn_cache.append({
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False
            })
        elif "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
            for _ in range(self.num_transformer_blocks):
                crossattn_cache.append({
                    "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                    "is_init": False
                })
        else:
            raise NotImplementedError

        self.crossattn_cache = crossattn_cache
