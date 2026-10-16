# file: features.py
# libraries: torch, transformers
# Configurable semantic metric backbones; their checkpoint identities are saved with results.
from __future__ import annotations
import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel, CLIPModel


class EvaluationFeatures(nn.Module):
    def __init__(self, clip_path: str, dino_path: str, video_path: str,
                 cache_dir: str, local_files_only: bool = False, chunk_size: int = 16) -> None:
        super().__init__()
        options = {'cache_dir': cache_dir, 'local_files_only': local_files_only}
        self.clip_processor = AutoImageProcessor.from_pretrained(clip_path, use_fast = False, **options)
        self.clip = CLIPModel.from_pretrained(clip_path, **options)
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_path, use_fast = False, **options)
        self.dino = AutoModel.from_pretrained(dino_path, **options)
        self.video_processor = AutoImageProcessor.from_pretrained(video_path, use_fast = False, **options)
        self.video = AutoModel.from_pretrained(video_path, **options)
        self.chunk_size = chunk_size
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> dict[str, torch.Tensor]:
        # metric processors consume RGB uint8; neither input is compressed again
        device = next(self.parameters()).device
        pixels = (videos.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 1, 3, 4, 2).numpy()
        clip, dino, video = [], [], []
        for frames in pixels:
            clip_frames, dino_frames = [], []
            for start in range(0, len(frames), self.chunk_size):
                images = list(frames[start:start + self.chunk_size])
                clip_input = self.clip_processor(images = images, return_tensors = 'pt').to(device)
                features = self.clip.get_image_features(**clip_input)
                clip_frames.append((features if isinstance(features, torch.Tensor) else features.pooler_output).cpu())
                dino_input = self.dino_processor(images = images, return_tensors = 'pt').to(device)
                dino_frames.append(self.dino(**dino_input).last_hidden_state[:, 0].cpu())
            clip.append(torch.cat(clip_frames))
            dino.append(torch.cat(dino_frames))
            # VideoMAE uses its configured number of uniformly spaced frames
            count = self.video.config.num_frames
            indices = torch.linspace(0, len(frames) - 1, count).round().long().tolist()
            inputs = self.video_processor([frames[i] for i in indices], return_tensors = 'pt').to(device)
            video.append(self.video(**inputs).last_hidden_state.mean(1).cpu())
        return {'clip': torch.stack(clip), 'dino': torch.stack(dino), 'video': torch.cat(video)}


class I3DFeatures(nn.Module):
    def __init__(self, path: str) -> None:
        super().__init__()
        # supply a trusted feature-only TorchScript I3D with forward(N,C,T,H,W) -> (N,D)
        self.model = torch.jit.load(path, map_location = 'cpu').eval()
        self.model.requires_grad_(False)

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
