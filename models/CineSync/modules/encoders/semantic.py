# file: semantic.py
# libraries: torch, transformers, pillow
# Frozen SigLIP2 teachers, matching the checkpoint named in the released loader.
# NOTE: this is the encoder of visual and textual features
#       their paper, p.g.7, under 5.1 Implementation Details
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor


class SemanticEncoder(nn.Module):
    def __init__(self, model_path: str = 'google/siglip2-so400m-patch14-384',
                 cache_dir: str | None = None, local_files_only: bool = False,
                 frame_batch_size: int = 16) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir = cache_dir, local_files_only = local_files_only, use_fast = False)
        self.model = AutoModel.from_pretrained(model_path, cache_dir = cache_dir, local_files_only = local_files_only)
        self.model.requires_grad_(False).eval()
        self.frame_batch_size = frame_batch_size

    def train(self, mode: bool = True) -> SemanticEncoder:
        # teachers remain deterministic even when the surrounding model enters training
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, videos: list, captions: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        frames = [frame for video in videos for frame in video]
        features = []
        # chunk teacher inference to bound memory while retaining all 33 frames
        for start in range(0, len(frames), self.frame_batch_size):
            inputs = self.processor.image_processor(images = frames[start:start + self.frame_batch_size], return_tensors = 'pt')
            inputs = inputs.to(device)
            value = self.model.get_image_features(**inputs)
            features.append(value if isinstance(value, torch.Tensor) else value.pooler_output)
        tokens = self.processor(text = captions, padding = 'max_length', truncation = True,
                                max_length = 64, return_tensors = 'pt').to(device)
        text = self.model.get_text_features(**tokens)
        text = text if isinstance(text, torch.Tensor) else text.pooler_output
        # the upstream joint SigLIP forward normalises both teacher embeddings
        visual = F.normalize(torch.cat(features).float(), dim = -1)
        return visual.reshape(len(videos), 33, -1), F.normalize(text.float(), dim = -1)
