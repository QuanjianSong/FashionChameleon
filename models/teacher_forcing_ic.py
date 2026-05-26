import torch.nn.functional as F
from typing import Tuple
import torch
from torch import nn

from backbones.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class TeacherForcingICModel(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__()
        self.args = args

        self.model_name = self.args.model_name
        self.is_causal = args.generator_type == "causal"
        self.dtype = torch.bfloat16 if self.args.mixed_precision else torch.float32
        self.device = device

        self._initialize_models()

        # initialize for causal
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

    def _initialize_models(self):
        self.generator = WanDiffusionWrapper(**getattr(self.args, "model_kwargs", {}),
                                            model_name=self.model_name,
                                            is_causal=self.is_causal,
                                            seq_list=list(self.args.image_or_video_shape),
                                            local_attn_size=self.args.local_attn_size,
                                        )
        self.generator.model.requires_grad_(True)
        if self.args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.text_encoder = WanTextEncoder(model_name=self.model_name)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_name=self.model_name)
        self.vae.requires_grad_(False)

        # initialize schedule
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
        if hasattr(self.args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(self.args.denoising_step_list, dtype=torch.long)
            if self.args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list].to(self.device)

    @torch.no_grad()
    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep

    def generator_loss(self, clean_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        batch_size, num_frame = clean_latent.shape[:2]

        noise = torch.randn_like(clean_latent)
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            batch_size,
            num_frame,
            self.num_frame_per_block,
            uniform_timestep=False,
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_input = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))
        training_target = self.scheduler.training_target(clean_latent, noise, timestep)

        clean_latent_aug = torch.cat([conditional_dict['src_latent'], conditional_dict['cloth_latent'], clean_latent], dim=1)    
        noisy_input = torch.cat([conditional_dict['src_latent'], conditional_dict['cloth_latent'], noisy_input], dim=1)

        if "2.2" in self.generator.model_name and '5B' in self.generator.model_name:
            temp_ts = F.pad(timestep, pad=(noisy_input.shape[1] - num_frame, 0), mode="constant", value=0)
            temp_ts = temp_ts[:, :, None, None].expand(-1, -1, list(self.args.image_or_video_shape)[-2] // 2, list(self.args.image_or_video_shape)[-1] // 2).to(device=self.device, dtype=self.dtype)
            temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
            wan22_input_timestep = temp_ts.to(self.device, dtype=torch.long)
        else:
            wan22_input_timestep = None
        # append anything
        conditional_dict['wan22_input_timestep'] = wan22_input_timestep

        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=F.pad(timestep, pad=(noisy_input.shape[1] - num_frame, 0), mode="constant", value=0),
            clean_x=clean_latent_aug,
            aug_t=None,
            in_context_nums=noisy_input.shape[1] - num_frame,
        )

        # compute the regression loss
        loss = torch.nn.functional.mse_loss(
            flow_pred[:, -num_frame:, :, :, :].float(), training_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, (batch_size, num_frame))
        loss = loss.mean()

        log_dict = {
            "x0": clean_latent_aug.detach(),
            "x0_pred": x0_pred.detach()
        }

        return loss, log_dict
