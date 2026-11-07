# file: gen_GT_clip_embeds.py
# this script generates OpenCLIP embeddings for the supplied Qwen captions
# run with `python -m dataset.cc2017.gen_GT_clip_embeds`
# optional preparation after get_cc2017.py; standard NEURONS uses the downloaded caption embeddings

import os
import torch
import numpy as np
from models.NEURONS.modules.generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder2 # bigG embedder from OpenCLIP
import json
from tqdm import tqdm



def gen_captions_embeds(mode, clip_embedder):
    qwen_captions = json.load(open(f'./dataset/cc2017/qwen_{mode}_caption_tag.json'))
    all_predcaptions = []
    all_predcaptions_emb = []
    for dict in tqdm(qwen_captions, ascii = True):
        caption = dict['caption']
        # print(f"\033[92m {caption} \033[0m")
        all_predcaptions = np.hstack((all_predcaptions, caption))

        with torch.no_grad():
            _, embedding = clip_embedder(caption)
        # print(f"\033[91m {embedding.shape} \033[0m")
        embedding = embedding.cpu()
        all_predcaptions_emb.append(embedding[0])


    print(f"\033[92m all_predcaptions {all_predcaptions.shape} \033[0m")
    torch.save(all_predcaptions, f'{root_dir}/GT_{mode}_caption_qwen.pt')

    all_predcaptions_emb = torch.stack(all_predcaptions_emb)
    print(f"\033[92m all_predcaptions_emb {all_predcaptions_emb.shape} \033[0m")
    torch.save(all_predcaptions_emb, f'{root_dir}/GT_{mode}_caption_qwen_emb.pt')


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir = './data/cc2017/qwen_annotation'
    os.makedirs(root_dir, exist_ok = True)

    clip_embedder = FrozenOpenCLIPEmbedder2(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        layer="last",
        legacy=False,
        always_return_pooled=True,
        cache_dir = "./data/cache/huggingface/hub"
    )
    clip_embedder.to(device).eval()


    gen_captions_embeds("train", clip_embedder)
    gen_captions_embeds("test", clip_embedder)
