
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from models.base import SelfForcingModel
from utils.loss import get_denoising_loss


class GradientReweightedDMDModel(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)

        # Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)
        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod
        else:
            self.scheduler.alphas_cumprod = None
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

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
            # breakpoint()
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

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        noisy_image_or_video = torch.cat([conditional_dict['src_data'], conditional_dict['cloth_data'], noisy_image_or_video], dim=1)
        if "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
            temp_ts = F.pad(timestep, pad=(conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1], 0), mode="constant", value=0)
            temp_ts = temp_ts[:, :, None, None].expand(-1, -1, list(self.args.image_or_video_shape)[-2] // 2, list(self.args.image_or_video_shape)[-1] // 2).to(device=self.device, dtype=self.dtype)
            temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
            #
            wan22_input_timestep = temp_ts.to(self.device, dtype=torch.long)
        else:
            wan22_input_timestep = None
        # append condition
        conditional_dict['wan22_input_timestep'] = wan22_input_timestep
        unconditional_dict['wan22_input_timestep'] = wan22_input_timestep

        # compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=F.pad(timestep, pad=(conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1], 0), mode="constant", value=0),
        )
        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=F.pad(timestep, pad=(conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1], 0), mode="constant", value=0)
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # compute the real score
        # We compute the conditional and unconditional prediction and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=F.pad(timestep, pad=(conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1], 0), mode="constant", value=0)
        )
        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=F.pad(timestep, pad=(conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1], 0), mode="constant", value=0)
        )
        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # remove in-context
        grad = grad[:, conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1]:, ...]
        pred_real_image = pred_real_image[:, conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1]:, ...]

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        pred_video_pixel: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        if pred_video_pixel is not None: # use gradient reweighted
            with torch.no_grad():
                pred_video_pixel = ((pred_video_pixel * 0.5 + 0.5) * 255).round().clamp(0, 255).to(torch.uint8)
                reward = self.reward_model(pred_video_pixel.flatten(0, 1)).unflatten(0, (batch_size, -1))
                # mean for latent
                reward = torch.cat((reward[:, :1], reward[:, 1:].view(batch_size, -1, 4).mean(dim=-1)), dim=-1)
                # mean for chunk
                reward = reward.view(batch_size, -1, self.num_frame_per_block).mean(dim=-1, keepdim=True).expand(-1, -1, self.num_frame_per_block).flatten(1, 2)
                temperature = -self.args.temperature
                # normalize to 0~1 (sum for 1)
                adavance = torch.nn.functional.softmax(reward / temperature, dim=-1)

        with torch.no_grad():
            # randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:
            if pred_video_pixel is not None:
                dmd_loss = 0.5 * (adavance * F.mse_loss(original_latent.double()[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="none").mean(dim=[2, 3, 4])).sum()
            else:
                dmd_loss = 0.5 * F.mse_loss(original_latent.double(
                )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            if pred_video_pixel is not None:
                dmd_loss = 0.5 * (adavance * F.mse_loss(original_latent.double(), (original_latent.double() - grad.double()).detach(), reduction="none").mean(dim=[2, 3, 4])).sum()
            else: 
                dmd_loss = 0.5 * F.mse_loss(original_latent.double(), (original_latent.double() - grad.double()).detach(), reduction="mean")

        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        use_gradient_reweighted=False,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to, pred_video_pixel = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
        )

        # compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            pred_video_pixel=pred_video_pixel if use_gradient_reweighted else None,
        )

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
            )

        # compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            generated_image.shape[0],
            generated_image.shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, generated_image.shape[:2])

        noisy_generated_image = torch.cat([conditional_dict['src_data'], conditional_dict['cloth_data'], noisy_generated_image], dim=1)

        if "2.2" in self.generator.model_name and "5B" in self.generator.model_name:
            temp_ts = F.pad(critic_timestep, pad=(conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1], 0), mode="constant", value=0)
            temp_ts = temp_ts[:, :, None, None].expand(-1, -1, list(self.args.image_or_video_shape)[-2] // 2, list(self.args.image_or_video_shape)[-1] // 2).to(device=self.device, dtype=self.dtype)
            temp_ts = temp_ts.reshape(temp_ts.shape[0], -1)
            wan22_input_timestep = temp_ts.to(self.device, dtype=torch.long)
        else:
            wan22_input_timestep = None
        # append condition
        conditional_dict['wan22_input_timestep'] = wan22_input_timestep

        flow_pred, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=F.pad(critic_timestep, pad=(conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1], 0), mode="constant", value=0)
        )

        # remove in-context
        flow_pred = flow_pred[:, conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1]: , ...]
        pred_fake_image = pred_fake_image[:, conditional_dict['src_data'].shape[1] + conditional_dict['cloth_data'].shape[1]: , ...]

        # compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            pred_fake_noise = None
        elif self.args.denoising_loss_type == "noise":
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])
        else:
            flow_pred = None
            pred_fake_noise = None

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred.flatten(0, 1),
        )

        # log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
