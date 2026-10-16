# file: cogvideo.py
# libraries: torch, diffusers == 0.35.2, transformers
# CogVideoX v1.0 backend; uses Diffusers weights rather than SAT checkpoints.
from __future__ import annotations
import torch
from torch import nn
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXDPMScheduler, CogVideoXPipeline
from .lora import attach_lora


class BrainVideoPipeline(CogVideoXPipeline):
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # the released VAE runs in float32 even when the diffusion transformer uses bfloat16
        with torch.autocast(device_type = latents.device.type, enabled = False):
            return super().decode_latents(latents.float())


class CogVideoBackend(nn.Module):
    def __init__(self, model_path: str, cache_dir: str | None = None,
                 local_files_only: bool = False, dtype: torch.dtype = torch.bfloat16,
                 rank: int = 64, alpha: float = 64, gradient_checkpointing: bool = True) -> None:
        super().__init__()
        options = {'cache_dir': cache_dir, 'local_files_only': local_files_only}
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder = 'transformer', torch_dtype = dtype, low_cpu_mem_usage = False, **options)
        self.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder = 'vae', torch_dtype = torch.float32, low_cpu_mem_usage = False, **options)
        self.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder = 'scheduler', **options)
        if self.transformer.config.patch_size_t is not None:
            raise ValueError('Use CogVideoX v1.0, not CogVideoX 1.5.')
        if self.scheduler.config.prediction_type != 'v_prediction':
            raise ValueError('Expected the CogVideoX v-prediction scheduler.')
        self.lora_targets = attach_lora(self.transformer, rank, alpha)
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.vae.requires_grad_(False).eval()
        self.vae.enable_slicing()
        self.vae.enable_tiling()
        # no T5 encoder is needed: brain tokens replace the text condition entirely
        self.pipeline = BrainVideoPipeline(tokenizer = None, text_encoder = None,
                                         transformer = self.transformer, vae = self.vae, scheduler = self.scheduler)

    def train(self, mode: bool = True) -> CogVideoBackend:
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # input B,T,C,H,W in [-1,1]; posterior sampling and 0.7 scaling match CogVideoX-5B
        with torch.autocast(device_type = video.device.type, enabled = False):
            latent = self.vae.encode(video.float().transpose(1, 2)).latent_dist.sample()
        return (latent * self.vae.config.scaling_factor).transpose(1, 2)

    def diffusion_loss(self, video: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        latents = self.encode_video(video)
        noise = torch.randn_like(latents)
        timestep = torch.randint(0, self.scheduler.config.num_train_timesteps, (len(latents),), device = latents.device)
        noisy = self.scheduler.add_noise(latents, noise, timestep)
        dtype = self.transformer.dtype
        rotary = self.pipeline._prepare_rotary_positional_embeddings(video.shape[-2], video.shape[-1], latents.shape[1], latents.device)
        prediction = self.transformer(hidden_states = noisy.to(dtype), encoder_hidden_states = condition.to(dtype),
                                      timestep = timestep, image_rotary_emb = rotary, return_dict = False)[0]
        # weighted x0 loss in upstream simplifies exactly to velocity MSE
        target = self.scheduler.get_velocity(latents, noise, timestep)
        return (prediction.float() - target.float()).square().mean()

    @torch.no_grad()
    def generate(self, condition: torch.Tensor, height: int = 480, width: int = 720,
                 steps: int = 50, guidance_scale: float = 6, generator: torch.Generator | None = None) -> torch.Tensor:
        self.eval()
        # zero brain conditioning supplies the unconditional classifier-free branch
        output = self.pipeline(prompt_embeds = condition.to(self.transformer.dtype),
                               negative_prompt_embeds = torch.zeros_like(condition).to(self.transformer.dtype),
                               num_frames = 33, height = height, width = width,
                               num_inference_steps = steps, guidance_scale = guidance_scale,
                               use_dynamic_cfg = True, generator = generator, output_type = 'pt')
        return output.frames
