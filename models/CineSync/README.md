# CineSync-fMRI

## Sources:

- [CineBrain paper, sections 4–5 and supplementary section 8.1](https://arxiv.org/html/2503.06940v2)
- [Released loader](https://github.com/yanweifu-sii/CineBrain/blob/main/data_video.py)
- [Released fMRI encoder](https://github.com/yanweifu-sii/CineBrain/blob/main/sgm/modules/encoders/fmri_encoder_custom.py)
- [Released conditioner and alignment losses](https://github.com/yanweifu-sii/CineBrain/blob/main/sgm/modules/encoders/modules.py)
- [Released diffusion configuration](https://github.com/yanweifu-sii/CineBrain/blob/main/configs/cogvideox_5b_lora_brain_va.yaml)
- [Released evaluation entry point](https://github.com/yanweifu-sii/CineBrain/blob/main/get_metric.py)

| Component | Implementation |
| --- | --- |
| Trial | 4 seconds, five fMRI frames, 33 inclusive video samples at nominal 8 Hz |
| fMRI | Visual ROI by default: `[:, :8405]` from `(5, 18946)`; auditory and combined ranges are configurable |
| Encoder | 12 pre-norm transformer layers, width 2048, 32 heads, 226 spatial tokens plus CLS; dropout 0.1 |
| Teacher | Frozen `google/siglip2-so400m-patch14-384`; 64 text tokens; normalised teacher embeddings of width 1152 |
| Video pooling | Trainable `Conv1d(33, 1, 1)` over frame embeddings |
| Stage one | Sum of symmetric fMRI/video and fMRI/text cross-entropy losses; learned scales initialised to `1/0.07` |
| Stage two | Trainable fMRI encoder and linear 2048→4096 projector; brain tokens replace all text conditioning |
| Video decoder | CogVideoX-5B **v1.0**, Diffusers-format model; frozen float32 VAE and frozen base transformer with attention LoRA |
| LoRA | Q/K/V/output projections, rank 64 and alpha 64, following the paper |
| Diffusion loss | Velocity MSE, algebraically equivalent to upstream weighted x0 MSE; 10% whole-condition dropout |
| Optimiser | AdamW, learning rate `1e-4`, betas `(0.9, 0.95)`; weight decay 0.01 is an explicit local choice |
| Schedule | Stage one: 50 epochs, batch 16, one GPU. Stage two: 5000 steps, batch 2 per GPU, four GPUs |
| Sampling | 33 frames, 480×720, 50 DPM steps, dynamic classifier-free guidance, scale 6 |

## Data:

`modules/dataset.py:Dataset_CineSync` subclasses **`dataset.dataset.Dataset` directly**. It opens LMDB separately in each worker, supports selected modalities and optional preloading, and maps captions by each JSON record's `video` filename. JSON list position is never used as a clip ID.

Before loading, a selected trial must indicate five fMRI frames or a four-second duration. Standard filenames such as `000005_0001_000005_0033.mp4` identify caption `000005.mp4`; crossing two source clips is rejected because a unique caption would be ambiguous. Model inputs additionally require five fMRI frames and 33 decoded video frames. No new haemodynamic shift or neural normalisation is applied.

Video paths are read directly from the indexed database entries. The base dataset's selections are respected; no automatic split is invented. Defaults train subject 1 using the database's `train` split and evaluate its `test` split. To reproduce per-subject results, run each subject separately. Stage two inherits its subject, ROI and encoder architecture from stage one's checkpoint.

## Installation and launch

Activate the project's environment, then install the small additional dependency
list:

```bash
pip install -r models/CineSync/requirements.txt
```

**Edit the literal configuration block in each Bash file.** AND Run the scripts from the **project root**:

```bash
bash models/CineSync/models/CineSync_fMRI/bash_scrips/train_align.sh
# Set ALIGN_CHECKPOINT in train_recon.sh to the stage-one checkpoint.
bash models/CineSync/models/CineSync_fMRI/bash_scrips/train_recon.sh
# Set CHECKPOINT in each evaluation launcher.
bash models/CineSync/models/CineSync_fMRI/bash_scrips/eval_align.sh
bash models/CineSync/models/CineSync_fMRI/bash_scrips/eval_recon.sh
```

Set `GPU_DEVICES="1,3,5,7"` and `NPROC=4` in the reconstruction launcher to select four GPUs. All training and evaluation launchers use `DEVICE_TYPE="cuda"` and export `CUDA_VISIBLE_DEVICES` from `GPU_DEVICES`; local ranks follow that list. For one GPU, set `GPU_DEVICES="2"` (and `NPROC=1` for reconstruction). Alignment and evaluation use one GPU. Python entry points also expose argparse for programmatic use and `--help`; Bash passes the configured values explicitly. `bash_scrips` is not a Python package.

All pretrained model identifiers can be replaced by local directories. Configure `CACHE_DIR`, `HF_ENDPOINT` and `LOCAL_FILES_ONLY` in the launcher. 

Reconstruction uses the official Diffusers `THUDM/CogVideoX-5b` model directory with `transformer`, `vae` and `scheduler` subfolders; it does not load T5 because brain tokens supply the condition.

## Run artefacts

```text
results/CineSync/CineSync_fMRI/<align|recon|eval_align|eval_recon>/run_<UTC timestamp>_<random suffix>/
├── <launch script>.sh
├── logs/                 # torch.utils.tensorboard events
├── checkpoints/          # periodic and final training checkpoints
├── meta/
│   ├── arguments.json    # resolved settings, including checkpoint paths
│   ├── model.json        # training architecture/stage metadata
│   └── metrics.json      # evaluation scores
└── videos/               # reconstruction evaluation only
```

Stage-two checkpoints store the brain encoder, projector and LoRA parameters, not including another copy of frozen 5B weights. They retain the base-model identifier and architecture needed for loading. `RESUME` restores weights, optimiser and progress; it restarts the saved epoch rather than guaranteeing bitwise-identical mid-epoch continuation. Keep the same base-model revision when resuming.

## Evaluation protocol

Alignment evaluation reports full-gallery cosine top-1/top-5 retrieval for brain -> video and brain -> text. Shared clip IDs count as positives. These diagnostic alignment metrics are not the paper's reconstruction N-way metrics.

Reconstruction evaluation supports:

- Frame PSNR and SSIM on matched RGB frames in `[0,1]`, before saving a compressed reconstruction MP4. Identical frames use a finite PSNR cap of 120 dB.
- Frame 2-way/50-way identification using configurable CLIP embeddings and video identification using configurable VideoMAE embeddings, 100 sampled distractor trials. At least N unique clips are required for an N-way score. A repeated clip contributes its first prediction to this gallery; pixel metrics retain all trials.
- DTC and CTC defined here as mean adjacent-frame cosine similarity of DINOv2 and CLIP features, respectively. The default model IDs and definitions are recorded; exact comparability requires the authors' missing metric helper/checkpoints.
- FVD from a **user-supplied feature-only TorchScript I3D**. Configure `I3D_TORCHSCRIPT`; the module must accept RGB `(N,3,T,224,224)` in `[-1,1]` and return pre-classification `(N,D)` features. No unrelated features are labelled as FVD. Without this file, the report explicitly omits FVD.

Evaluation uses the database's indexed video clips for GT `SAVE_ARRAYS=true` additionally preserves float reconstructions for exact later scoring. Semantic metric models and their cache locations are configurable.

## Learning-rate scheduling

Both training launchers expose `LR_DECAY_STYLE`, `LR_DECAY_STEPS`, `LR_WARMUP`, `LR_DECAY_RATIO`, `LR_AUTO_WARMUP_STEPS` and `LR_AUTO_WARMUP_RATE`. The decay horizon is inferred from epochs, loader length, gradient accumulation and `MAX_STEPS` unless specified explicitly. Scheduling advances once per optimiser update;

These defaults follow the released [SAT arguments](https://github.com/yanweifu-sii/CineBrain/blob/main/sat/arguments.py) and [learning-rate scheduler](https://github.com/yanweifu-sii/CineBrain/blob/main/sat/training/learning_rates.py):

1% linear warm-up, linear decay, and an initial fine-tuning cap of 5% of the base learning rate for 100 updates. Set `LR_AUTO_WARMUP_STEPS=0` to disable the cap. Cosine decay uses `LR_DECAY_RATIO` as its minimum multiplier; constant keeps the base rate after warm-up. As in SAT, decay uses the full horizon after warm-up, so its endpoint is reached at warm-up plus the configured horizon. Linear decay is clamped at zero if training continues beyond that endpoint. Zero warm-up is supported without division by zero.

The repository's references is absent (`sft_5b_brain_va_clip_sub05.yaml`)
