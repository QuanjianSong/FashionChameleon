from backbones.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
import torch
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from tqdm import tqdm


class CausalWan22ICInferencePipeline(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.timestep_shift = getattr(args, "timestep_shift", 5.0)

        # initialize all models
        self.generator_model_name = getattr(args, 'model_name')
        self.generator = WanDiffusionWrapper(
            **getattr(self.args, "model_kwargs", {}),
            model_name=self.generator_model_name,
            is_causal=True,
            seq_list=list(self.args.image_or_video_shape),
            local_attn_size=self.args.local_attn_size,
            sink_size=2 + self.args.num_frame_per_block, # in-context + sink
        )
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
        elif "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
            self.num_transformer_blocks = 30
            self.frame_seq_length = 880
            self.num_frame_per_block = getattr(args, 'num_frame_per_block')
            self.context_noise = 0

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = getattr(args, 'independent_first_frame')
        self.num_max_frames = getattr(args, 'num_training_frames')
        self.kv_cache_size = self.num_max_frames * self.frame_seq_length

    @torch.no_grad()
    def causal_inference(
        self,
        noisy_image_or_video,
        text_prompts,
        src_data=None,
        cloth_data=None,
    ):
        batch_size, num_frames, num_channels, height, width = noisy_image_or_video.shape

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        if src_data is not None and cloth_data is not None:
            initial_latent = torch.cat([src_data, cloth_data], dim=1)
        else:
            initial_latent = None

        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

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

        # # pre-filling for in-context frame
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, num_input_frames], device=noisy_image_or_video.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :num_input_frames] = initial_latent

            temp_ts = timestep[:, :, None, None].expand(-1, -1, height // 2, width // 2).to(device=initial_latent.device, dtype=initial_latent.dtype)
            temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
            wan22_input_timestep = temp_ts.to(initial_latent.device, dtype=torch.long)

            # append anything
            conditional_dict['wan22_input_timestep'] = wan22_input_timestep
            unconditional_dict['wan22_input_timestep'] = wan22_input_timestep

            # condition
            self.generator(
                noisy_image_or_video=initial_latent,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length
            )
            # uncondition
            self.generator(
                noisy_image_or_video=initial_latent,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length
            )
            current_start_frame += num_input_frames

        # temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for idx, current_num_frames in enumerate(tqdm(all_num_frames)):
            noisy_input = noisy_image_or_video[
                :, current_start_frame - num_input_frames: current_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            sample_scheduler = self._initialize_sample_scheduler(noisy_image_or_video)

            # spatial denoising loop
            for index, current_timestep in enumerate(sample_scheduler.timesteps):
                latent_model_input = latents
                # print(f"current_timestep: {current_timestep}")

                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noisy_image_or_video.device,
                    dtype=torch.int64) * current_timestep

                if "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
                    temp_ts = timestep[:, :, None, None].expand(-1, -1, latents.shape[-2] // 2, latents.shape[-1] // 2)
                    temp_ts = temp_ts.reshape(temp_ts.shape[0], -1) # torch.Size([1, 15004])
                    wan22_input_timestep = temp_ts.to(noisy_image_or_video.device, dtype=torch.long)
                else:
                    wan22_input_timestep = None
                
                # append condition
                conditional_dict['wan22_input_timestep'] = wan22_input_timestep
                unconditional_dict['wan22_input_timestep'] = wan22_input_timestep

                # condition
                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length
                )
                # uncondition
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length
                )
                
                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    current_timestep,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0

            # record the model's output
            output[:, current_start_frame: current_start_frame + current_num_frames] = latents

            # rerun with timestep zero to update KV cache using clean context
            context_timestep = self.context_noise * torch.ones(
                    [batch_size, current_num_frames], device=noisy_image_or_video.device, dtype=torch.long)

            if "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
                temp_ts = context_timestep[:, :, None, None].expand(-1, -1, latents.shape[-2] // 2, latents.shape[-1] // 2)
                temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
                wan22_input_timestep = temp_ts.to(noisy_image_or_video.device, dtype=torch.long)
            else:
                wan22_input_timestep = None
            # append condition
            conditional_dict['wan22_input_timestep'] = wan22_input_timestep
            unconditional_dict['wan22_input_timestep'] = wan22_input_timestep

            # condition
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
            )
            # uncondition
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if src_data is not None and cloth_data is not None:
            output = output[:, num_input_frames:]

        video = self.vae.decode_to_pixel(output)

        video = (video * 0.5 + 0.5).clamp(0, 1)

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

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_pos = []
        kv_cache_neg = []

        if "2.1" in self.generator.model_name and "1.3B" in self.generator.model_name:
            for _ in range(self.num_transformer_blocks):
                kv_cache_pos.append({
                    "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })
                kv_cache_neg.append({
                    "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })
        elif "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
            for _ in range(self.num_transformer_blocks):
                kv_cache_pos.append({
                    "k": torch.zeros([batch_size, self.kv_cache_size, 24, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, self.kv_cache_size, 24, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })
                kv_cache_neg.append({
                    "k": torch.zeros([batch_size, self.kv_cache_size, 24, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, self.kv_cache_size, 24, 128], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })
        else:
            raise NotImplementedError

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        #
        if "2.1" in self.generator.model_name and "1.3B" in self.generator.model_name:
            for _ in range(self.num_transformer_blocks):
                crossattn_cache_pos.append({
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False
                })
                crossattn_cache_neg.append({
                    "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                    "is_init": False
                })
        elif "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
            for _ in range(self.num_transformer_blocks):
                crossattn_cache_pos.append({
                    "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                    "is_init": False
                })
                crossattn_cache_neg.append({
                    "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                    "is_init": False
                })
        else:
            raise NotImplementedError

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache
