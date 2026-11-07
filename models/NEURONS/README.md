# NEURONS

Adapted from [xmed-lab/NEURONS](https://github.com/xmed-lab/NEURONS/tree/4e4fd545d982747cc9e2a4af2eb0186a22eb5c84), retaining the official architecture, losses, raw cc2017 dataset and stage ordering. Shared building blocks live in `modules/`; the official variant, configurations and entry points live in `models/NEURONS/`.

> [!NOTE]
> Compatibility adjustments support the project's existing environment, and TensorBoard records scalars and segmentation images.

## Data and installation

Activate the project's environment and run from the **project root**:

```bash
pip install -r models/NEURONS/requirements.txt
python -m dataset.cc2017.get_cc2017
python -m dataset.cc2017.get_NEURONSassets
python -m dataset.cc2017.find_key_obj
```

Edit the setup blocks in the download scripts as needed. Raw data, annotations and pretrained weights go under `data/cc2017`; library caches go under `data/cache`. **NOT supporting our universal dataset for now (the lmdb-based override)**

> [!IMPORTANT]
> Upstream currently lacks the required `masks/*` and `qwen_annotation/*` downloads; obtain them from the authors, place them under `data/cc2017`, extract the mask archives and run `find_key_obj` before either training stage.

`get_NEURONSassets`: Set `NEURONS_PATTERNS` to optionally download published experiment checkpoints under `data/cc2017/NEURONS`, then copy the selected checkpoint into your experiment's `checkpoints/` directory for reconstruction.

`python -m dataset.cc2017.gen_GT_clip_embeds` optionally generates embeddings for the bundled Qwen captions; standard training uses the caption embeddings supplied with cc2017.

## Launch

**Edit the literal configuration block** in `models/NEURONS/models/NEURONS/train_neurons.sh`, then run from the **project root**:

```bash
bash models/NEURONS/models/NEURONS/train_neurons.sh
```

`STAGES` selects any combination, executed in the following order:

| Stage | Operation |
| --- | --- |
| `1` | Train the brain backbone |
| `2` | Train the NEURONS decoupler |
| `3` | Reconstruct keyframes and blurry video |
| `4` | Caption reconstructed keyframes with BLIP2 |
| `5` | Generate final videos with AnimateDiff |
| `6` | Evaluate generated videos |
| `e` | Export decoupled outputs |

Configure GPU IDs, subject, paths and training settings in Bash. Training and keyframe stages use one process; stage 5 uses the GPUs selected by `GPU_DEVICES`. Video diffusion settings are in `VIDEO_CONFIG` and its referenced inference YAML. `MODE` supports upstream `enhance`, `motion` and `self` combinations; stage 4 is optional for modes containing `self`.

## Run artefacts

Outputs default to `results/NEURONS/NEURONS/exp_<name>/subj_<id>`, containing checkpoints, stage logs, TensorBoard events and generated artefacts. Python entry points accept `--exp_dir` to override this directory; video generation uses `--exp` for the full directory path.

Stage 2 requires stage 1's `checkpoints/brain_model.pth`; reconstruction requires `checkpoints/brain_model_prior_last.pth`.

```bash
tensorboard --logdir results/NEURONS/NEURONS
```
