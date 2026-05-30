from typing import Tuple
from torch import nn
import torch
import torch.distributed as dist

from backbones.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from pipelines import SelfForcingWan22ICTrainingPipeline
from utils.aesthetic_scorer import AestheticScorer


class SelfForcingModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args

        self.model_name = self.args.model_name
        self.is_causal = args.generator_type == "causal"
        self.dtype = torch.bfloat16 if self.args.mixed_precision else torch.float32
        self.device = device

        self._initialize_models()

        # additional for causal
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 23)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        self.inference_pipeline = None

    def _initialize_models(self):
        # if dist.get_rank() == 0:
        #     breakpoint()
        # dist.barrier()
        if getattr(self.args, "use_gradient_reweighted", False):
            self.reward_model = AestheticScorer(dtype=torch.float32)
            self.reward_model.requires_grad_(False)

        self.generator = WanDiffusionWrapper(
                                            **getattr(self.args, "model_kwargs", {}),
                                            model_name=self.model_name,
                                            is_causal=self.is_causal,
                                            seq_list=list(self.args.image_or_video_shape),
                                            local_attn_size=self.args.local_attn_size,
                                            sink_size=getattr(self.args, "sink_size", 0),    
                                        )
    
        self.generator.model.requires_grad_(True)
        if self.args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.real_score = WanDiffusionWrapper(
                                            model_name=self.model_name,
                                            is_causal=False,
                                            seq_list=list(self.args.image_or_video_shape),
                                        )
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(
                                            model_name=self.model_name,
                                            is_causal=False,
                                            seq_list=list(self.args.image_or_video_shape),
                                        )
        self.fake_score.model.requires_grad_(True)
        if self.args.gradient_checkpointing:
            self.fake_score.enable_gradient_checkpointing()

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

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"

        noise_shape = image_or_video_shape.copy()

        # we have 2 frames as in-context
        assert (self.num_training_frames - 2) % self.num_frame_per_block == 0
        assert (self.num_training_frames - 2) % self.num_frame_per_block == 0
        max_num_blocks = (self.num_training_frames - 2) // self.num_frame_per_block
        min_num_blocks = (self.num_training_frames - 2) // self.num_frame_per_block

        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block

        # sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noisy_image_or_video=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
            **conditional_dict,
        )

        # remove the in-context frames
        pred_image_or_video_last = pred_image_or_video[:, (self.num_training_frames - num_generated_frames):, ...].to(self.dtype)
        # no use mask
        gradient_mask = None

        if getattr(self.args, 'use_gradient_reweighted', False):
            with torch.no_grad():
                pred_video_pixel = self.vae.decode_to_pixel(pred_image_or_video_last).to(torch.float32)
        else:
            pred_video_pixel = None

        return pred_image_or_video_last, gradient_mask, denoised_timestep_from, denoised_timestep_to, pred_video_pixel

    def _consistency_backward_simulation(
        self,
        noisy_image_or_video: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noisy_image_or_video=noisy_image_or_video, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        if self.is_causal:
            self.inference_pipeline = SelfForcingWan22ICTrainingPipeline(
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=self.args.independent_first_frame,
                same_step_across_blocks=self.args.same_step_across_blocks,
                last_step_only=self.args.last_step_only,
                num_max_frames=self.num_training_frames,
                context_noise=self.args.context_noise,
            )
        else:
            raise NotImplementedError
