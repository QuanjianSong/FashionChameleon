from backbones.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
from utils.scheduler import SchedulerInterface
from typing import List
import torch
import torch.distributed as dist


class SelfForcingWan22ICTrainingPipeline:
    def __init__(self,
                denoising_step_list: List[int],
                scheduler: SchedulerInterface,
                generator: WanDiffusionWrapper,
                vae: WanVAEWrapper = None,
                num_frame_per_block: int = 3,
                independent_first_frame: bool = False,
                same_step_across_blocks: bool = False,
                last_step_only: bool = False,
                num_max_frames: int = 21,
                context_noise: int = 0,
                **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.vae = vae
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # wan specific hyperparameters
        if "2.1" in self.generator.model_name and "1.3B" in self.generator.model_name:
            self.num_transformer_blocks = 30
            self.frame_seq_length = 1560
            self.num_frame_per_block = num_frame_per_block
            self.context_noise = context_noise
            self.i2v = False
        elif "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
            self.num_transformer_blocks = 30
            self.frame_seq_length = 880
            self.num_frame_per_block = num_frame_per_block
            self.context_noise = context_noise
            self.i2v = False
        else:
            raise NotImplementedError

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.num_max_frames = num_max_frames

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
        self,
        noisy_image_or_video: torch.Tensor,
        return_sim_step: bool = False,
        **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noisy_image_or_video.shape

        # in-context frames
        initial_latent = torch.cat([conditional_dict['src_data'], conditional_dict['cloth_data']], dim=1)

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
            batch_size=batch_size, dtype=noisy_image_or_video.dtype, device=noisy_image_or_video.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noisy_image_or_video.dtype, device=noisy_image_or_video.device
        )

        # cache in-context feature
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
            with torch.no_grad():
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
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noisy_image_or_video.device)
        start_gradient_frame_index = num_output_frames - self.num_max_frames

        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noisy_image_or_video[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            # spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # only backprop at the randomly selected timestep (consistent across all ranks)
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

                if not exit_flag:
                    with torch.no_grad():
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
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            # record the model's output
            output[:, current_start_frame: current_start_frame + current_num_frames] = denoised_pred

            # get clean timestep
            context_timestep = self.context_noise * torch.ones(
                    [batch_size, current_num_frames], device=noisy_image_or_video.device, dtype=torch.long)
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])
            if "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
                temp_ts = context_timestep[:, :, None, None].expand(-1, -1, height // 2, width // 2).to(device=noisy_input.device, dtype=noisy_input.dtype)
                temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
                wan22_input_timestep = temp_ts.to(noisy_image_or_video.device, dtype=torch.long)
            else:
                wan22_input_timestep = None
            # append condition
            conditional_dict['wan22_input_timestep'] = wan22_input_timestep
            # pre-filling to get kv
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            # update the start and end frame indices
            current_start_frame += current_num_frames

        # return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    @torch.no_grad()
    def validate(
        self,
        noisy_image_or_video: torch.Tensor,
        **conditional_dict,
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noisy_image_or_video.shape

        # in-context frames
        initial_latent = torch.cat([conditional_dict['src_data'], conditional_dict['cloth_data']], dim=1)

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
            with torch.no_grad():
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

        for current_num_frames in all_num_frames:
            noisy_input = noisy_image_or_video[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
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

            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep.flatten(0, 1),
            ).unflatten(0, denoised_pred.shape[:2])

            if "2.2" in self.generator.model_name and "5B" in self.generator.model_name:                
                temp_ts = context_timestep[:, :, None, None].expand(-1, -1, height // 2, width // 2).to(device=noisy_input.device, dtype=noisy_input.dtype)
                temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
                wan22_input_timestep = temp_ts.to(noisy_image_or_video.device, dtype=torch.long)
            else:
                wan22_input_timestep = None
            # append anything
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

        return output

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
