"""
DualForce Diffusion Forcing Scheduler.

Extends FlowMatchScheduler with per-frame independent noise levels,
which is the core of Diffusion Forcing for autoregressive generation.

Key difference from standard flow matching:
- Standard: one sigma per sample (shared across all frames)
- Diffusion Forcing: independent sigma per frame, enabling:
  - Training: each frame gets a random noise level
  - Inference: sliding window denoising for autoregressive generation
"""

import torch
import math
from typing import Optional, Tuple

from mova.diffusion.schedulers.flow_match import FlowMatchScheduler


class DiffusionForcingScheduler(FlowMatchScheduler):
    """
    Flow matching scheduler with per-frame independent noise levels.

    Supports two noise ranges for dual-modality (video + 3D struct):
    - sigma_v: video noise range [0, sigma_v_max]
    - sigma_s: struct noise range [0, sigma_s_max] (typically smaller for stability)
    """

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=5.0,
        sigma_max=1.0,
        sigma_min=0.0,
        # Diffusion Forcing specific
        sigma_v_max: float = 1.0,       # Video noise range max
        sigma_s_max: float = 0.7,       # Struct noise range max (more stable)
        noise_sampling: str = "uniform", # "uniform" or "logit_normal"
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        # Parent args
        inverse_timesteps=False,
        extra_one_step=True,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            inverse_timesteps=inverse_timesteps,
            extra_one_step=extra_one_step,
            reverse_sigmas=reverse_sigmas,
            exponential_shift=exponential_shift,
            exponential_shift_mu=exponential_shift_mu,
            shift_terminal=shift_terminal,
        )
        self.sigma_v_max = sigma_v_max
        self.sigma_s_max = sigma_s_max
        self.noise_sampling = noise_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std

    def sample_per_frame_sigma(
        self,
        batch_size: int,
        num_frames: int,
        sigma_max: float = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Sample independent noise levels for each frame.

        Args:
            batch_size: B
            num_frames: T (number of temporal frames)
            sigma_max: maximum sigma value (default: self.sigma_v_max)
            device: target device

        Returns:
            sigmas: [B, T] per-frame noise levels in [0, sigma_max]
        """
        if sigma_max is None:
            sigma_max = self.sigma_v_max

        if self.noise_sampling == "uniform":
            # Uniform sampling in [0, sigma_max]
            sigmas = torch.rand(batch_size, num_frames, device=device) * sigma_max

        elif self.noise_sampling == "logit_normal":
            # Logit-normal distribution (SD3-style) scaled to [0, sigma_max]
            u = torch.zeros(batch_size, num_frames, device=device)
            torch.nn.init.trunc_normal_(
                u, mean=self.logit_mean, std=self.logit_std,
                a=torch.logit(torch.tensor(0.001)).item(),
                b=torch.logit(torch.tensor(0.999)).item(),
            )
            sigmas = torch.sigmoid(u) * sigma_max

        else:
            raise ValueError(f"Unknown noise sampling: {self.noise_sampling}")

        return sigmas

    def sample_dual_per_frame_sigma(
        self,
        batch_size: int,
        num_frames_video: int,
        num_frames_struct: int,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample independent noise levels for both video and struct modalities.

        Args:
            batch_size: B
            num_frames_video: T_v (video temporal frames)
            num_frames_struct: T_s (struct temporal frames, usually same as video)
            device: target device

        Returns:
            sigma_v: [B, T_v] video per-frame noise levels
            sigma_s: [B, T_s] struct per-frame noise levels
        """
        sigma_v = self.sample_per_frame_sigma(
            batch_size, num_frames_video, self.sigma_v_max, device)
        sigma_s = self.sample_per_frame_sigma(
            batch_size, num_frames_struct, self.sigma_s_max, device)
        return sigma_v, sigma_s

    def add_noise_per_frame(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise with per-frame sigma levels.

        Args:
            original_samples: [B, C, T, H, W] for video or [B, D, T] for struct
            noise: same shape as original_samples
            sigmas: [B, T] per-frame noise levels

        Returns:
            noisy_samples: same shape as original_samples
        """
        ndim = original_samples.dim()

        if ndim == 5:
            # Video: [B, C, T, H, W] -> expand sigma to [B, 1, T, 1, 1]
            sigma_expanded = sigmas[:, None, :, None, None]
        elif ndim == 3:
            # Struct: [B, D, T] -> expand sigma to [B, 1, T]
            sigma_expanded = sigmas[:, None, :]
        else:
            raise ValueError(f"Unexpected tensor dim: {ndim}")

        # Flow matching interpolation: x_t = (1 - sigma) * x_0 + sigma * noise
        noisy_samples = (1 - sigma_expanded) * original_samples + sigma_expanded * noise
        return noisy_samples

    def training_target_per_frame(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute v-prediction target for flow matching.

        target = noise - sample (independent of sigma)

        Args:
            original_samples: clean data
            noise: noise samples

        Returns:
            target: v-prediction target
        """
        return noise - original_samples

    def sigma_to_timestep(self, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Convert sigma values to timestep values (for AdaLN conditioning).

        Args:
            sigmas: [B, T] or [B] sigma values

        Returns:
            timesteps: same shape, in [0, num_train_timesteps] range
        """
        return sigmas * self.num_train_timesteps

    # --------------------------------------------------
    # Inference helpers for autoregressive generation
    # --------------------------------------------------

    def get_sliding_window_sigmas(
        self,
        total_frames: int,
        window_size: int,
        overlap: int,
        num_denoise_steps: int = 25,
    ) -> torch.Tensor:
        """
        Generate sigma schedule for sliding window autoregressive inference.

        For each window position, returns a denoising schedule where:
        - New frames start at sigma_max and are denoised to sigma_min
        - Previously denoised frames have sigma=0 (or small re-noise)
        - Overlap frames have intermediate sigma for smooth blending

        Args:
            total_frames: total number of frames to generate
            window_size: number of frames per denoising window
            overlap: number of overlapping frames between windows
            num_denoise_steps: denoising steps per window

        Returns:
            schedule: list of (frame_indices, sigma_schedule) tuples
        """
        stride = window_size - overlap
        schedule = []

        for start in range(0, total_frames, stride):
            end = min(start + window_size, total_frames)
            frame_indices = list(range(start, end))

            # Create per-frame sigma schedule for this window
            window_sigmas = torch.zeros(num_denoise_steps, end - start)

            for step in range(num_denoise_steps):
                t = step / max(num_denoise_steps - 1, 1)
                sigma = self.sigma_v_max * (1 - t)  # Linear denoising

                for i, frame_idx in enumerate(frame_indices):
                    if frame_idx < start:
                        # Already denoised frame: no noise
                        window_sigmas[step, i] = 0.0
                    elif frame_idx < start + overlap and start > 0:
                        # Overlap frame: partial noise (blend)
                        blend = (frame_idx - start) / overlap
                        window_sigmas[step, i] = sigma * blend
                    else:
                        # New frame: full denoising
                        window_sigmas[step, i] = sigma

            schedule.append((frame_indices, window_sigmas))

            if end >= total_frames:
                break

        return schedule
