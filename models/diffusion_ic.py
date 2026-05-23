import torch.nn.functional as F
from typing import Tuple
import torch
from torch import nn

from backbones.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class DiffusionICModel(nn.Module):
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
        self.generator = WanDiffusionWrapper(
                                            **getattr(self.args, "model_kwargs", {}),
                                            model_name=self.model_name,
                                            seq_list=list(self.args.image_or_video_shape),
                                            local_attn_size=self.args.local_attn_size,
                                            is_causal=self.is_causal,
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

    def generator_loss(self, clean_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        noise = torch.randn_like(clean_latent)
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (clean_latent.shape[0], 1)).expand(-1, clean_latent.shape[1])
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.dtype, device=self.device)
        
        noisy_input = self.scheduler.add_noise(
            clean_latent.flatten(0, 1), 
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, clean_latent.shape[:2])
        v_target = self.scheduler.training_target(clean_latent, noise, timestep)

        noisy_input = torch.cat([conditional_dict['src_data'], conditional_dict['cloth_data'], noisy_input], dim=1)

        if "2.2" in self.generator.model_name and '5B' in self.generator.model_name:
            temp_ts = F.pad(timestep, pad=(noisy_input.shape[1] - clean_latent.shape[1], 0), mode="constant", value=0)
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
            timestep=F.pad(timestep, pad=(noisy_input.shape[1] - clean_latent.shape[1], 0), mode="constant", value=0),
        )

        # compute the regression loss
        loss = torch.nn.functional.mse_loss(
            flow_pred[:, -v_target.shape[1]:, :, :, :].float(), v_target.float(), reduction='none'
        ).mean(dim=(2, 3, 4))

        loss = loss * self.scheduler.training_weight(timestep).unflatten(0, clean_latent.shape[:2])
        loss = loss.mean()

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach()
        }

        return loss, log_dict
