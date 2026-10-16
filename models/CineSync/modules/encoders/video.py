# file: video.py
# libraries: torch, torchvision, numpy, av
from __future__ import annotations
import numpy as np
import torch
from torchvision.transforms.functional import resize, center_crop, InterpolationMode


def prepare_video(videos: list[np.ndarray], height: int = 480, width: int = 720) -> torch.Tensor:
    # preserve aspect ratio, then use the upstream centre crop and [-1,1] normalisation
    result = []
    for video in videos:
        frames = torch.from_numpy(video.copy()).permute(0, 3, 1, 2)
        h, w = frames.shape[-2:]
        scale = max(height / h, width / w)
        frames = resize(frames, [max(height, int(h * scale)), max(width, int(w * scale))],
                        interpolation = InterpolationMode.BICUBIC, antialias = True)
        frames = center_crop(frames, [height, width])
        result.append(frames.float() / 127.5 - 1)
    return torch.stack(result)


def save_video(path: str, video: torch.Tensor, fps: int = 8) -> None:
    import av
    # video is T,C,H,W in [0,1]; metrics use the in-memory output before encoding
    frames = (video.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1) * 255).round().to(torch.uint8).numpy()
    with av.open(path, mode = 'w') as container:
        stream = container.add_stream('libx264', rate = fps)
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18'}
        for pixels in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format = 'rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
