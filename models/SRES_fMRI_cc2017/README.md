# SRES_fMRI_cc2017

denoise/deconvolve structure and reblurring objective generally follows [ZS-DeconvNet](https://github.com/TristaZeng/ZS-DeconvNet/tree/main/Python_MATLAB_Codes)

> [!IMPORTANT]
> Now for temporal signals !!! \
> ZS-DeconvNet uses Spatial optical convolutions (modelled by PSF) on non-negative fluorescence outputs and spatial Hessian losses. \
> As causal HRF models temporal relations of BOLD response, the model works on time scale now. \
> There is also a sPSF on blood flow dynamics or so, which is possible to use too.

## Data audit

The loader reads `[trial, repetition, voxel]` tensors and extracts chronological windows without normalisation. The [published tensor description](https://huggingface.co/datasets/gongzx/cc2017_dataset) identifies 13,447/14,828/9,114 selected voxels for subjects 1/2/3; it supplies no documented voxel-coordinate file. **These flattened vectors do not define a regular spatial grid.**

The defaults declare 18 training and five testing movie runs, each containing 240 two-second TRs. Runs 17 and 18 of the training set are held out for validation. Windows never cross movie or repetition boundaries; only complete windows are shuffled. Training and calibration preserve individual repeats, while inference can average them.

> [!NOTE]
> This should not lead to data leakage. But noted that NEURONS will not further split the training set portion for validation. \
> This model will calibrate ONLY on the training set (excluding the validation portion)

The default configuration assumes a four-second (`STORED_DELAY_SEC`) BOLD delay already applied in the saved stimulus pairing. (I do think this is true for cc2017, as NEURONS used the data sequence directly)

Source values, chronological order and per-run z-scoring could not be verified. No extra normalisation is introduced.

## Training (in general)

Configure and run:

```bash
bash models/SRES_fMRI_cc2017/models/v1/run.sh
```

Set `ACTION="calibrate"` to estimate voxel noise on training runs, then `ACTION="train"` and `STAGE=1` for denoising. Set `STAGE=2` and `STAGE1_CHECKPOINT` to the resulting `checkpoints/last.pt` for synthetic pre-training followed by real-data adaptation. Stage 2 fine-tunes both branches unless `FREEZE_DENOISER=true`. `RESUME` restores model, optimiser, scheduler and Torch RNG states with matching training settings.

The 32-TR window, 15-TR burn-in and four-TR trailing exclusion accommodate the full default 30-second causal HRF. Increase the window/burn-in when extending kernel support. Both U-Nets use offline temporal context; `LOOKAHEAD_TRS` excludes trailing output positions and is not a causal receptive-field limit. A full-context neural decoder cannot be assumed to operate online.

> [!NOTE]
> Offline means processing recordings that are already available. The model can use both past and future samples to estimate activity at a particular time. \
> Online means processing data as it arrives. A strictly causal model can use only current and past measurements; using future measurements requires waiting.

## Noise and forward model

Calibration uses within-run first differences: `sigma_v = MAD(diff(Y_v)) / (0.9539 * sqrt(1 - rho_v))`. `RHO_AR1="auto"` estimates per-voxel AR(1) coefficients from paired training-repeat differences; it requires at least two presentations. Noise starts in its stationary distribution, with innovation scale `sqrt(1 - rho_v**2)`. Fixed alpha and sigma generate `input = Y + alpha*z` and `target = Y - z/alpha` with fresh draws each iteration.

> [!NOTE]
> R2R decorrelation relies on an adequate scanner-noise covariance model: temporal differences may contain stimulus changes, repeat differences assume repeat stability, and the default model omits spatial noise covariance.

Stage 1 minimises interior de-noising MSE. Stage 2 minimises `mu * de-noising_MSE + (1 - mu) * re-blurred_MSE + lambda_tv * mean(abs(diff(Z)) / dt)`. Re-blurring flips the double-gamma HRF kernel, pads only on the left, convolves in the time domain and samples every `TSR_FACTOR` points. Losses exclude incomplete HRF history; FFT is not proposed (HRF runs on temporal not frequency domain). A bounded learnable dilation is optional, and signed activity plus a learnable voxel baseline is enabled by default. Baseline and latent DC activity are not separately identifiable without additional assumptions.

Synthetic pre-training supervises known block activity using HRF-generated observations with extra leading history and calibrated noise. Its amplitude and shot durations are configurable. It initialises the de-convolver; real-data losses subsequently use no clean activation labels. Temporal upsampling provides a regularised hypothesis, not proof that sub-TR neural timing is recoverable.

**sPSF**: Spatial blur is disabled by default (`ENABLE_SPATIAL_SPSF`). An optional `SPATIAL_MATRIX_PATH` Torch dictionary must contain a non-negative row-normalised `[V,V]` `matrix`, its `fwhm_mm`, and `voxel_order_confirmed=True`. Construct this distance-decay operator from verified coordinates in exactly the tensor voxel order; a sparse matrix avoids dense quadratic storage. **This is spatial de-blurring on existing voxels, not spatial super-resolution onto a new grid.** Spatial blur and optional low-rank voxel mixing require all voxels together during training.

## Inference

Set `ACTION="infer"`, `CHECKPOINT` and `MODE="A"` or `"B"`. Keep architecture, physics and noise-amplitude settings consistent with the checkpoint. Monte-Carlo inference averages re-corrupted predictions and overlapping valid window interiors. Averaging repetitions scales calibrated noise by `1/sqrt(repeats)`, assuming independent repeat noise. Results include NaN boundary positions and an explicit validity mask; downstream consumers must exclude invalid positions.

Mode A exports de-noised BOLD as `[stimulus_trial, repetition, voxel]`, retaining the original delayed pairing. Mode B exports inferred activity as `[stimulus_trial, repetition, subframe, voxel]`: stored acquisition index `i` is assigned to stimulus index `i + stored_delay_sec/TR`, within the same run. This removes the assumed pairing delay without shifting the forward convolution twice. Valid samples retain their original GT trial indices.

Evaluation records de-noising-change MSE and, for mode B, re-blurred-input MSE as input-consistency diagnostics rather than clean-signal reconstruction accuracy.

## Run artefacts

Runs are saved under `results/SRES_fMRI_cc2017/v1/<action>_stage<N>/run_<timestamp>_<id>/`, including the launcher, resolved arguments, data audit, metrics, TensorBoard logs and atomic `last.pt`/`best.pt` checkpoints. Inference writes enhanced tensors/outputs and timing/validity metadata under `artifacts/`.
