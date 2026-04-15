import math
from typing import Union

import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class WanStepDistillScheduler(WanScheduler):
    def __init__(self, config):
        # WanSchedulerのdim/num_headsが不要なためBaseSchedulerのみ呼ぶ
        BaseScheduler.__init__(self, config)

        if config.get("denoising_step_list"):
            # 明示的に指定されている場合はそのまま使う
            self.denoising_step_list = config["denoising_step_list"]
        else:
            # infer_stepsから自動計算（ComfyUI simple schedulerと同等のタイムステップ）
            infer_steps = config["infer_steps"]
            step = 1000 // infer_steps
            self.denoising_step_list = [1000 - i for i in range(0, 1000, step)][:infer_steps]

        self.infer_steps = len(self.denoising_step_list)
        self.sample_shift = self.config["sample_shift"]
        self.noise_pred = None

        self.num_train_timesteps = 1000
        self.sigma_max = 1.0
        self.sigma_min = 0.0

    def prepare_latents(self, seed, latent_shape, dtype=torch.float32):
        # ComfyUI版と一致させるためCPU generatorでnoise生成後GPUに転送
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.latents = torch.randn(
            latent_shape[0],
            latent_shape[1],
            latent_shape[2],
            latent_shape[3],
            dtype=dtype,
            device="cpu",
            generator=self.generator,
        ).to(AI_DEVICE)
        print(f"[PREPARE_LATENTS] seed={seed}, latents std={self.latents.float().std():.4f}, shape={list(self.latents.shape)}")

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        if image_encoder_output is not None and "vae_encoder_out" in image_encoder_output:
            self.vae_encoder_out = image_encoder_output["vae_encoder_out"]
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)
        self.set_denoising_timesteps(device=AI_DEVICE)

    def set_denoising_timesteps(self, device: Union[str, torch.device] = None):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, self.num_train_timesteps + 1)[:-1]
        self.sigmas = self.sample_shift * self.sigmas / (1 + (self.sample_shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

        self.denoising_step_index = [self.num_train_timesteps - x for x in self.denoising_step_list]
        self.timesteps = self.timesteps[self.denoising_step_index].to(device)
        self.sigmas = self.sigmas[self.denoising_step_index].to("cpu")

    def reset(self, seed, latent_shape, step_index=None):
        self.prepare_latents(seed, latent_shape, dtype=torch.float32)

    def add_noise(self, original_samples, noise, sigma):
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def step_post(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        sample_before = self.latents.float().std().item()
        latents_f32 = self.latents.to(torch.float32)

        # sigma_n: 次ステップのsigma（最終ステップは0）
        sigma_n = self.sigmas[self.step_index + 1].item() if self.step_index < self.infer_steps - 1 else 0.0
        # Euler相当: latents + flow_pred * (sigma_n - sigma)
        noisy_image_or_video = latents_f32 + flow_pred * (sigma_n - sigma)

        self.latents = noisy_image_or_video.to(self.latents.dtype)
        print(f"[SCHEDULER_DISTILL] step{self.step_index}: latents_before std={sample_before:.4f}, flow_pred mean={flow_pred.float().mean():.4f}, std={flow_pred.float().std():.4f}, sigma={sigma:.4f}, latents_after std={self.latents.float().std():.4f}")
        if self.step_index == self.infer_steps - 1:
            print(f"[FINAL_LATENT] latents shape={list(self.latents.shape)}, std={self.latents.float().std():.4f}")


class Wan21MeanFlowStepDistillScheduler(WanStepDistillScheduler):
    def __init__(self, config):
        super().__init__(config)

    def step_pre(self, step_index):
        super().step_pre(step_index)
        self.timestep_input = torch.stack([self.timesteps[self.step_index]])
        if self.config["model_cls"] == "wan2.2" and self.config["task"] in ["i2v", "s2v", "rs2v"]:
            self.timestep_input = (self.mask[0][:, ::2, ::2] * self.timestep_input).flatten()
        if self.config["model_cls"] == "wan2.1_mean_flow_distill":
            t_next = self.timesteps[self.step_index + 1] if self.step_index < self.infer_steps - 1 else torch.zeros_like(self.timestep_input)
            self.timestep_input_r = torch.stack([t_next])


class Wan22StepDistillScheduler(WanStepDistillScheduler):
    def __init__(self, config):
        super().__init__(config)
        self.boundary_step_index = config["boundary_step_index"]

    def set_denoising_timesteps(self, device: Union[str, torch.device] = None):
        super().set_denoising_timesteps(device)
        self.sigma_bound = self.sigmas[self.boundary_step_index].item()

    def calculate_alpha_beta_high(self, sigma):
        alpha = (1 - sigma) / (1 - self.sigma_bound)
        beta = math.sqrt(sigma**2 - (alpha * self.sigma_bound) ** 2)
        return alpha, beta

    def step_post(self):
        flow_pred = self.noise_pred.to(torch.float32)
        sigma = self.sigmas[self.step_index].item()
        noisy_image_or_video = self.latents.to(torch.float32) - flow_pred * sigma
        if self.step_index < self.infer_steps - 1:
            sigma_n = self.sigmas[self.step_index + 1].item()
            noisy_image_or_video = noisy_image_or_video + flow_pred * sigma_n
        self.latents = noisy_image_or_video.to(self.latents.dtype)
