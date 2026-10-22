# file: features.py
# libraries: torch, transformers
# Configurable semantic metric backbones; their checkpoint identities are saved with results.
from __future__ import annotations
import torch
from torch import nn
from transformers import (AutoImageProcessor, AutoModel, AutoProcessor, CLIPVisionModelWithProjection,
                          ViTImageProcessor, ViTForImageClassification,
                          VideoMAEImageProcessor, VideoMAEForVideoClassification)


class EvaluationFeatures(nn.Module):
    def __init__(self, clip_path: str, dino_path: str, video_path: str, image_path: str,
                 cache_dir: str, local_files_only: bool = False, chunk_size: int = 16) -> None:
        super().__init__()
        options = {'cache_dir': cache_dir, 'local_files_only': local_files_only}
        self.clip_processor = AutoProcessor.from_pretrained(clip_path, use_fast = False, **options)
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_path, **options)
        self.image_processor = ViTImageProcessor.from_pretrained(image_path, **options)
        self.image = ViTForImageClassification.from_pretrained(image_path, **options)
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_path, use_fast = False, **options)
        self.dino = AutoModel.from_pretrained(dino_path, **options)
        self.video_processor = VideoMAEImageProcessor.from_pretrained(video_path, **options)
        # get_metric.py passes all 33 frames, overriding the pretrained 16-frame configuration
        self.video = VideoMAEForVideoClassification.from_pretrained(video_path, num_frames = 33, **options)
        self.chunk_size = chunk_size
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> dict[str, torch.Tensor]:
        # metric processors consume RGB uint8; neither input is compressed again
        device = next(self.parameters()).device
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        for model in (self.clip, self.image, self.video):
            if model.dtype != dtype:
                model.to(dtype = dtype)
        pixels = (videos.detach().cpu().clamp(0, 1) * 255).to(torch.uint8).permute(0, 1, 3, 4, 2).numpy()
        if videos.shape[1] != self.video.config.num_frames:
            raise ValueError('CineBrain reconstruction metrics require 33 video frames.')
        clip, dino, image_logits, video_logits = [], [], [], []
        for frames in pixels:
            clip_frames, dino_frames, image_frames = [], [], []
            for start in range(0, len(frames), self.chunk_size):
                images = list(frames[start:start + self.chunk_size])
                # NOTE: CineSync issue #3 uses half precision for CLIP, ViT and VideoMAE on CUDA
                clip_input = self.clip_processor(images = images, return_tensors = 'pt').to(device, self.clip.dtype)
                clip_frames.append(self.clip(**clip_input).image_embeds.float().cpu())
                image_input = self.image_processor(images = images, return_tensors = 'pt').to(device, self.image.dtype)
                image_frames.append(self.image(**image_input).logits.cpu())
                dino_input = self.dino_processor(images = images, return_tensors = 'pt').to(device)
                dino_frames.append(self.dino(**dino_input).last_hidden_state[:, 0].cpu())
            clip.append(torch.cat(clip_frames))
            dino.append(torch.cat(dino_frames))
            image_logits.append(torch.cat(image_frames))
            # entire video processed at once
            inputs = self.video_processor(list(frames), return_tensors = 'pt').to(device, self.video.dtype)
            video_logits.append(self.video(**inputs).logits.cpu())
        return {'clip': torch.stack(clip), 'dino': torch.stack(dino),
                'image_logits': torch.stack(image_logits), 'video_logits': torch.cat(video_logits)}


class I3DFeatures(nn.Module):
    def __init__(self, path: str) -> None:
        super().__init__()
        # supply a trusted feature-only TorchScript I3D with forward(N,C,T,H,W) -> (N,D)
        self.model = torch.jit.load(path, map_location = 'cpu').eval()
        # freeze parameters directly; TorchScript does not support module.requires_grad_
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = videos.shape
        x = torch.nn.functional.interpolate(videos.reshape(b * t, c, h, w), size = (224, 224),
                                            mode = 'bilinear', align_corners = False)
        x = (x.reshape(b, t, c, 224, 224).transpose(1, 2) * 2 - 1).contiguous()
        result = self.model(x)
        if not isinstance(result, torch.Tensor) or result.ndim != 2:
            raise ValueError('I3D feature checkpoint must return N,D embeddings, not classifier logits or a dictionary.')
        return result.cpu()
