# file: gen_database.py
# create the CineBrain database from the files extracted by get_CineBrain.py
# run with `python -m dataset.CineBrain.gen_database`
# dependencies: numpy, lmdb, av (PyAV provides the MP4 decoder and encoder)
# No additional filtering, normalisation or 4-second delay is applied by default.
# NOTE: UP - we assumed that CineBrain processed that for us

from __future__ import annotations

# [SET UP INFO] ==========================
DATASET_PATH = "./data/CineBrain"
DATAFOLDER_PATH = "./data/CineBrain_database"
DATABASE_PATH = f"{DATAFOLDER_PATH}/lmdb"
# Database resources (disk files) are stored in DATAFOLDER_PATH
DATABASE_NAME = "CineBrain"
DATABASE_MAP_SIZE = int(1e12)
SUBJECT_IDS = (1, 2, 3, 4, 5, 6)    # subjects to process (1-6)
SUBJECT_DIRECTORY = "sub-{subject_id:04d}"
EEG_DIRECTORY = "eeg_02"
FMRI_DIRECTORY = "visual_audio"
VIDEO_DIRECTORY = "clips"
BEHAVIOUR_DIRECTORY = "sub{subject_id:02d}_beh"
BEHAVIOUR_FILE = "sub{subject_id:02d}_{run}.csv"
NEURAL_FILE = "{index}.npy"
VIDEO_FILE = "{index:06d}.mp4"

# The paper specifies 20 runs of 18 minutes per subject, with 270 clips per run.
# Its split is runs 1-18 for training and 19-20 for testing (no validation split).
# https://arxiv.org/html/2503.06940v2 (sections 3.1 and 3.2)
RUNS_PER_SUBJECT = 20
EPISODE_DURATION_SECONDS = 18 * 60
FMRI_TR_SECONDS = 0.8
CLIP_DURATION_SECONDS = 4.0
# NOTE: this is variable, set to determine how long a trails should be
# NOTE: the maximum is no more than all frames WITHIN a run (i.e. an episode's duration)
FMRI_FRAMES_PER_TRIAL = 5  # 5 = 4 s; 20 = 16 s; 2 = 1.6 s; 3 = 2.4 s
TRAIN_RUNS = tuple(range(1, 19))
VAL_RUNS = ()
TEST_RUNS = (19, 20)
TRIAL_STATE = "task"
EEG_CHANNEL_COUNT = 64
EEG_SAMPLING_RATE = 1000
FMRI_VOXEL_COUNT = 18946

# The released JSON annotations used by BrainDataset are not publicly included.
# Infer neural numbering as subject-local, zero-based, contiguous 1350-frame
# runs. Infer clip numbering as S07E01-10, S09E01-10, S11E01-10, 270 clips each.
# CSVs establish episode identities, but cannot prove either numbering scheme.
VIDEO_EPISODE_ORDER = tuple((season, episode) for season in (7, 9, 11) for episode in range(1, 11))
SUBJECT_SEASONS = {1: (7, 9), 2: (7, 9), 3: (7, 11), 4: (7, 11), 5: (7, 11), 6: (7, 9)}
EPISODES_PER_SEASON = 10
NEURAL_FIRST_INDEX = 0
VIDEO_FIRST_INDEX = 0
# NOTE: following run-starts are the first number of the sequence
#       if episode A starts from 000010.mp4 to 000019.mp4, the start of A is 10
#       Override if needed, our code will by default assign automatically.
# Keys are (subject_id, run); values are the first numbered neural file.
NEURAL_RUN_STARTS: dict[tuple[int, int], int] = {}
# Keys are (season, episode); values are the first numbered video clip.
VIDEO_EPISODE_STARTS: dict[tuple[int, int], int] = {}
# Use this only for absent/blank behavioural episode identities, e.g. sub01 run 10.
ALLOW_SCHEDULE_FALLBACK = True

# Identical hook API for EEG and fMRI: ordered functions taking/returning arrays.
# Hooks run ONCE PER TRIAL, after concatenation/stacking, never per 0.8 s file.
# Example: [partial(normalise, axis = -1)] for EEG or axis = 0 for fMRI.
# For EEG downsampling, also set EEG_OUTPUT_SAMPLING_RATE to the hook's new rate.
# Long-recording ICA should be performed separately before trial generation.
EEG_PREPROCESSING_FUNCTIONS = None
FMRI_PREPROCESSING_FUNCTIONS = None
EEG_OUTPUT_SAMPLING_RATE = EEG_SAMPLING_RATE

# Upstream data_video.py reads 33 frames at nominal 8 Hz per four-second clip.
# Treat these as endpoints-inclusive samples at 0, 1/8, ..., 4 s. Adjacent clips
# share a boundary timestamp; merged trials keep only one frame at each boundary.
# For fractional 1/8-second ends, store the exact endpoint as an extra sample.
# This timing interpretation is explicit because the annotation JSON is absent.
VIDEO_SAMPLING_RATE = 8
SOURCE_FRAMES_PER_CLIP = 33
VIDEO_CACHE_CLIPS = 2
# H.264 compression reduces disk usage; decoded pixels may differ from the source.
# High CRF means more compression and lower quality; low CRF means less compression and higher quality.
VIDEO_CRF = 18
VIDEO_PRESET = "medium"

# Prompt if total OR any subject/run loses more than this fraction of its frames.
# Loss includes incomplete tails, missing EEG/fMRI and missing video dependencies.
# Non-interactive input aborts at the threshold; nothing is silently accepted.
DISCARD_FRACTION_THRESHOLD = 0.05
PROGRESS_EVERY_TRIALS = 100
# =======================================

# import libraries
import csv
import json
import math
import re
from collections import Counter, OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from dataset.database import Database
from dataset.CineBrain.EEG_Helper import load_eeg, preprocess_eeg
from dataset.CineBrain.fMRI_Helper import load_fmri, preprocess_fmri


# logger
def log(message: str, level: str = "INFO") -> None:
    colours = {"INFO": 34, "WARN": 33, "ERROR": 31, "OK": 32}
    print(f"\033[1;{colours[level]}m[CineBrain] {level}\033[0m: {message}", flush = True)


def integer_ratio(numerator: float, denominator: float, name: str) -> int:
    # refuse fractional neural frames or EEG samples rather than silently rounding
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"Invalid timing parameters for {name}.")
    ratio = numerator / denominator
    if ratio <= 0 or not math.isclose(ratio, round(ratio), rel_tol = 0, abs_tol = 1e-8):
        raise ValueError(f"{name} must be a positive integer, got {ratio}.")
    return int(round(ratio))


def validate_setup() -> dict[str, int]:
    """Validate configuration before scanning files or creating a database."""
    # calculate all derived counts from the timing parameters in the setup section
    counts = {
        "frames_per_run": integer_ratio(EPISODE_DURATION_SECONDS, FMRI_TR_SECONDS, "Frames per run"),
        "frames_per_clip": integer_ratio(CLIP_DURATION_SECONDS, FMRI_TR_SECONDS, "Frames per clip"),
        "clips_per_run": integer_ratio(EPISODE_DURATION_SECONDS, CLIP_DURATION_SECONDS, "Clips per run"),
        "samples_per_frame": integer_ratio(FMRI_TR_SECONDS * EEG_SAMPLING_RATE, 1, "EEG samples per frame"),
        "source_intervals": integer_ratio(CLIP_DURATION_SECONDS * VIDEO_SAMPLING_RATE, 1, "Video intervals per clip"),
    }
    for name, value in (("FMRI_FRAMES_PER_TRIAL", FMRI_FRAMES_PER_TRIAL),
                        ("VIDEO_CACHE_CLIPS", VIDEO_CACHE_CLIPS),
                        ("PROGRESS_EVERY_TRIALS", PROGRESS_EVERY_TRIALS),
                        ("EEG_CHANNEL_COUNT", EEG_CHANNEL_COUNT),
                        ("FMRI_VOXEL_COUNT", FMRI_VOXEL_COUNT)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if FMRI_FRAMES_PER_TRIAL > counts["frames_per_run"]:
        raise ValueError("A trial cannot be longer than an episode.")
    integer_ratio(FMRI_FRAMES_PER_TRIAL * FMRI_TR_SECONDS * EEG_OUTPUT_SAMPLING_RATE, 1, "Output EEG samples per trial")
    if SOURCE_FRAMES_PER_CLIP != counts["source_intervals"] + 1:
        raise ValueError("The configured video frame count must include both clip endpoints.")
    if not 0 <= DISCARD_FRACTION_THRESHOLD <= 1:
        raise ValueError("DISCARD_FRACTION_THRESHOLD must be between zero and one.")
    if not SUBJECT_IDS or len(set(SUBJECT_IDS)) != len(SUBJECT_IDS):
        raise ValueError("SUBJECT_IDS must be non-empty and unique.")
    if len(set(VIDEO_EPISODE_ORDER)) != len(VIDEO_EPISODE_ORDER):
        raise ValueError("VIDEO_EPISODE_ORDER contains duplicate episodes.")

    # ensure that every run belongs to exactly one phase
    runs = list(TRAIN_RUNS) + list(VAL_RUNS) + list(TEST_RUNS)
    if sorted(runs) != list(range(1, RUNS_PER_SUBJECT + 1)):
        raise ValueError("Train, validation and test runs must partition all subject runs exactly once.")
    if not Path(DATASET_PATH).is_dir():
        raise FileNotFoundError(f"Dataset directory '{DATASET_PATH}' does not exist.")
    if Path(DATABASE_PATH).exists():
        raise FileExistsError(f"Database '{DATABASE_PATH}' already exists; choose a new output directory.")
    return counts


def episode_identity(subject_id: int, run: int) -> tuple[tuple[int, int], str]:
    """Read SxxExx from all non-empty stim_path rows in the behavioural CSV."""
    path = Path(DATASET_PATH) / BEHAVIOUR_DIRECTORY.format(subject_id = subject_id)
    path = path / BEHAVIOUR_FILE.format(subject_id = subject_id, run = run)
    episodes: set[tuple[int, int]] = set()
    if path.is_file():
        # some CSVs have a trailing summary row or reordered columns
        with path.open(newline = "", encoding = "utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                stimulus = row.get("stim_path") or ""
                match = re.search(r"S(\d+)E(\d+)", stimulus, flags = re.IGNORECASE)
                if match:
                    episodes.add((int(match[1]), int(match[2])))
                elif stimulus.strip():
                    raise ValueError(f"Cannot interpret stimulus path in '{path}': {stimulus}")
    if len(episodes) > 1:
        raise ValueError(f"Conflicting episode identities in '{path}': {episodes}")
    if episodes:
        return episodes.pop(), str(path)

    # the released sub01_10.csv has no stimulus identity; use the declared schedule
    if not ALLOW_SCHEDULE_FALLBACK:
        raise ValueError(f"Missing episode identity in '{path}'; schedule fallback is disabled.")
    seasons = SUBJECT_SEASONS.get(subject_id, ())
    season_index, episode_index = divmod(run - 1, EPISODES_PER_SEASON)
    if season_index >= len(seasons):
        raise ValueError(f"No fallback schedule for subject {subject_id}, run {run}.")
    episode = (seasons[season_index], episode_index + 1)
    log(f"Subject {subject_id}, run {run}: missing CSV identity; inferred S{episode[0]:02d}E{episode[1]:02d} from the published schedule.", "WARN")
    return episode, "published_schedule_fallback"


def numbered_files(directory: Path, suffix: str, filename_template: str) -> set[int]:
    # inspect filenames only; missing indices must remain holes in the timeline
    return {int(path.stem) for path in directory.glob(f"*{suffix}")
            if path.is_file() and path.stem.isdigit()
            and path.name == filename_template.format(index = int(path.stem))}


def phase_for_run(run: int) -> str:
    # use the published split unless the setup explicitly supplies validation runs
    if run in TRAIN_RUNS:
        return "train"
    return "val" if run in VAL_RUNS else "test"


def video_sample_mapping(start_frame: int, counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return episode-local clip indices, source frame indices and trial times."""
    # sample a half-open interval, then retain its exact final endpoint once
    duration = FMRI_FRAMES_PER_TRIAL * FMRI_TR_SECONDS
    times = np.arange(int(math.ceil(duration * VIDEO_SAMPLING_RATE - 1e-8)), dtype = np.float64) / VIDEO_SAMPLING_RATE
    times = np.append(times, duration)
    absolute_times = start_frame * FMRI_TR_SECONDS + times
    clip_indices = np.floor(absolute_times / CLIP_DURATION_SECONDS + 1e-8).astype(np.int64)

    # at the trial endpoint use the previous clip's last frame, avoiding a dependency
    # on the following clip (which might belong to a different episode)
    if math.isclose(absolute_times[-1] / CLIP_DURATION_SECONDS, round(absolute_times[-1] / CLIP_DURATION_SECONDS), rel_tol = 0, abs_tol = 1e-8):
        clip_indices[-1] -= 1
    local_times = absolute_times - clip_indices * CLIP_DURATION_SECONDS
    source_indices = np.floor(local_times * VIDEO_SAMPLING_RATE + 0.5).astype(np.int64)
    source_indices = np.clip(source_indices, 0, SOURCE_FRAMES_PER_CLIP - 1)
    if np.any(clip_indices < 0) or np.any(clip_indices >= counts["clips_per_run"]):
        raise ValueError("A video trial crosses an episode boundary.")
    return clip_indices, source_indices, times


def make_plan(counts: dict[str, int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Plan complete, aligned trials and count losses before writing any payloads."""
    root = Path(DATASET_PATH)
    videos = numbered_files(root / VIDEO_DIRECTORY, ".mp4", VIDEO_FILE)
    plan: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    episode_phases: dict[tuple[int, int], str] = {}
    expected_frames = counts["frames_per_run"]
    expected_total = len(SUBJECT_IDS) * RUNS_PER_SUBJECT * expected_frames
    starts = {episode: VIDEO_FIRST_INDEX + i * counts["clips_per_run"]
              for i, episode in enumerate(VIDEO_EPISODE_ORDER)}
    starts.update(VIDEO_EPISODE_STARTS)

    # disallow reused clip ranges even when an override maps two episodes together
    ordered_starts = sorted(starts.values())
    if any(end - begin < counts["clips_per_run"] for begin, end in zip(ordered_starts, ordered_starts[1:])):
        raise ValueError("Configured video episode ranges overlap.")

    for subject_id in SUBJECT_IDS:
        # compare complete expected runs, including subjects whose files are all absent
        subject_path = root / SUBJECT_DIRECTORY.format(subject_id = subject_id)
        eeg_files = numbered_files(subject_path / EEG_DIRECTORY, ".npy", NEURAL_FILE)
        fmri_files = numbered_files(subject_path / FMRI_DIRECTORY, ".npy", NEURAL_FILE)
        seen_episodes: set[tuple[int, int]] = set()
        neural_ranges: list[tuple[int, int]] = []
        for run in range(1, RUNS_PER_SUBJECT + 1):
            episode, identity_source = episode_identity(subject_id, run)
            if episode in seen_episodes:
                raise ValueError(f"Subject {subject_id} has a duplicate episode: {episode}.")
            seen_episodes.add(episode)
            if episode not in starts:
                raise ValueError(f"No video numbering configured for episode {episode}.")
            phase = phase_for_run(run)
            previous_phase = episode_phases.setdefault(episode, phase)
            if previous_phase != phase:
                raise ValueError(f"Episode {episode} would appear in both {previous_phase} and {phase} across subjects.")
            neural_start = NEURAL_RUN_STARTS.get((subject_id, run), NEURAL_FIRST_INDEX + (run - 1) * expected_frames)
            if neural_start < 0 or starts[episode] < 0:
                raise ValueError("File indices must be non-negative.")
            if any(neural_start < end and neural_start + expected_frames > begin for begin, end in neural_ranges):
                raise ValueError(f"Overlapping neural run indices for subject {subject_id}, run {run}.")
            neural_ranges.append((neural_start, neural_start + expected_frames))
            complete_end = expected_frames - expected_frames % FMRI_FRAMES_PER_TRIAL
            losses: Counter[str] = Counter()
            losses["episode_tail"] = expected_frames - complete_end
            retained = 0

            for offset in range(0, complete_end, FMRI_FRAMES_PER_TRIAL):
                # never join across a gap: drop the entire fixed-grid trial if incomplete
                indices = range(neural_start + offset, neural_start + offset + FMRI_FRAMES_PER_TRIAL)
                missing_eeg = any(index not in eeg_files for index in indices)
                missing_fmri = any(index not in fmri_files for index in indices)
                local_clips, _, _ = video_sample_mapping(offset, counts)
                clip_ids = sorted({starts[episode] + int(index) for index in local_clips})
                missing_video = any(index not in videos for index in clip_ids)
                reasons = [name for name, missing in (("missing_eeg", missing_eeg), ("missing_fmri", missing_fmri), ("missing_video", missing_video)) if missing]
                if reasons:
                    # combined reason keys avoid counting the same discarded frame twice
                    losses["+".join(reasons)] += FMRI_FRAMES_PER_TRIAL
                    continue
                plan.append({
                    "subject_id": subject_id, "run": run, "season": episode[0],
                    "episode": episode[1], "phase": phase, "frame_offset": offset,
                    "neural_start": neural_start + offset, "video_start": starts[episode],
                    "first_clip_id": clip_ids[0], "last_clip_id": clip_ids[-1],
                    "identity_source": identity_source,
                })
                retained += FMRI_FRAMES_PER_TRIAL

            # record per-run losses so a missing episode cannot hide in a global average
            discarded = expected_frames - retained
            summaries.append({
                "subject_id": subject_id, "run": run, "season": episode[0], "episode": episode[1],
                "phase": phase, "identity_source": identity_source,
                "expected_frames": expected_frames, "retained_frames": retained,
                "discarded_frames": discarded, "discarded_fraction": discarded / expected_frames,
                "reasons": dict(losses), "neural_start": neural_start, "video_start": starts[episode],
            })
            log(f"Subject {subject_id:04d}, run {run:02d}, S{episode[0]:02d}E{episode[1]:02d} ({phase}): {retained // FMRI_FRAMES_PER_TRIAL} trials, {discarded}/{expected_frames} frames discarded.")

    retained_total = len(plan) * FMRI_FRAMES_PER_TRIAL
    report = {
        "expected_frames": expected_total, "retained_frames": retained_total,
        "discarded_frames": expected_total - retained_total,
        "discarded_fraction": (expected_total - retained_total) / expected_total,
        "trial_count": len(plan), "runs": summaries,
    }
    return plan, report


def confirm_discard(report: dict[str, Any]) -> None:
    # report total losses and request a decision before the database is created
    log(f"Total: {report['trial_count']} trials; discarding {report['discarded_frames']}/{report['expected_frames']} neural frames ({report['discarded_fraction']:.2%}).")
    if report["trial_count"] == 0:
        raise RuntimeError("No complete trials remain; check the dataset paths and numbering assumptions.")
    exceeded = [run for run in report["runs"] if run["discarded_fraction"] > DISCARD_FRACTION_THRESHOLD]
    if report["discarded_fraction"] > DISCARD_FRACTION_THRESHOLD or exceeded:
        log(f"Discard threshold {DISCARD_FRACTION_THRESHOLD:.2%} exceeded globally or in {len(exceeded)} run(s). See the generation report for reasons.", "WARN")
        try:
            answer = input("\033[1;33m[CineBrain] WARN\033[0m: Continue with these discarded frames? (y/n): ")
        except EOFError as exc:
            raise RuntimeError("Discard confirmation requires interactive input; database creation aborted.") from exc
        if answer.strip().lower() != "y":
            raise RuntimeError("Database creation aborted by the user.")


def decode_clip(path: Path) -> np.ndarray:
    """
    Decode source RGB pixels for selecting and joining trial frames.
    """

    # import PyAV library
    import av

    # fully decode the short source clip, preserving pixel values and spatial dimensions
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format = "rgb24") for frame in container.decode(video = 0)]
    if len(frames) != SOURCE_FRAMES_PER_CLIP:
        raise ValueError(f"'{path}' has {len(frames)} frames; expected {SOURCE_FRAMES_PER_CLIP}. Check the source video timing configuration.")
    return np.stack(frames, axis = 0)


def load_trial_video(trial: dict[str, Any], counts: dict[str, int], cache: OrderedDict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    # derive both cuts and joins from the same neural-frame interval used in planning
    local_clips, frame_indices, times = video_sample_mapping(trial["frame_offset"], counts)
    clip_indices = local_clips + trial["video_start"]
    pieces = []
    for index in np.unique(clip_indices):
        index = int(index)
        if index not in cache:
            path = Path(DATASET_PATH) / VIDEO_DIRECTORY / VIDEO_FILE.format(index = index)
            cache[index] = decode_clip(path)
            while len(cache) > VIDEO_CACHE_CLIPS:
                cache.popitem(last = False)
        cache.move_to_end(index)
        pieces.append(cache[index][frame_indices[clip_indices == index]])
    return np.concatenate(pieces, axis = 0), times


def encode_video(path: Path, video: np.ndarray, times: np.ndarray) -> None:
    """
    Save the selected RGB frames as a compressed MP4 with trial timestamps.
    """
    
    # import PyAV library
    import av

    # create exclusively so existing clips cannot be overwritten by another run
    with path.open("xb") as handle:
        try:
            with av.open(handle, mode = "w", format = "mp4") as container:
                stream = container.add_stream("libx264", rate = Fraction(str(VIDEO_SAMPLING_RATE)))
                stream.width = video.shape[2]
                stream.height = video.shape[1]
                # retain odd spatial dimensions without cropping or padding
                stream.pix_fmt = "yuv420p" if stream.width % 2 == 0 and stream.height % 2 == 0 else "yuv444p"
                stream.options = {"crf": str(VIDEO_CRF), "preset": VIDEO_PRESET}
                stream.time_base = Fraction(1, 1000000)
                stream.codec_context.time_base = stream.time_base

                # preserve fractional final intervals as well as both inclusive endpoints
                for pixels, timestamp in zip(video, times):
                    frame = av.VideoFrame.from_ndarray(pixels, format = "rgb24")
                    frame.time_base = stream.codec_context.time_base
                    frame.pts = round(float(timestamp) / frame.time_base)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        except BaseException:
            # remove only the incomplete file created by this call
            path.unlink(missing_ok = True)
            raise


def save_trial_video(
        trial: dict[str, Any], counts: dict[str, int], clips_path: Path,
        cache: OrderedDict[int, np.ndarray], saved_clips: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], np.ndarray]:
    # name each segment using original clip IDs and one-based, inclusive frame numbers
    local_clips, frame_indices, times = video_sample_mapping(trial["frame_offset"], counts)
    first_clip = int(local_clips[0]) + trial["video_start"]
    last_clip = int(local_clips[-1]) + trial["video_start"]
    filename = f"{first_clip:06d}_{int(frame_indices[0]) + 1:04d}_{last_clip:06d}_{int(frame_indices[-1]) + 1:04d}.mp4"

    # subjects sharing a stimulus segment reference the same generated file
    if filename not in saved_clips:
        video, times = load_trial_video(trial, counts, cache)
        path = clips_path / filename
        encode_video(path, video, times)
        saved_clips[filename] = {
            "storage": "disk", "path": str(path.resolve()), "type": "mp4",
            "offset_index": 0, "shape": tuple(video.shape), "dtype": str(video.dtype),
        }
    return saved_clips[filename], times


def load_trial_neural(trial: dict[str, Any], counts: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    # load matched EEG segments and fMRI frames without a second haemodynamic delay
    subject_path = Path(DATASET_PATH) / SUBJECT_DIRECTORY.format(subject_id = trial["subject_id"])
    eeg_segments = []
    fmri_frames = []
    for index in range(trial["neural_start"], trial["neural_start"] + FMRI_FRAMES_PER_TRIAL):
        filename = NEURAL_FILE.format(index = index)
        eeg = load_eeg(subject_path / EEG_DIRECTORY / filename, eeg_channel_count = EEG_CHANNEL_COUNT)
        fmri = load_fmri(subject_path / FMRI_DIRECTORY / filename)
        if eeg.shape != (EEG_CHANNEL_COUNT, counts["samples_per_frame"]):
            raise ValueError(f"Unexpected EEG shape for subject {trial['subject_id']}, file {filename}: {eeg.shape}.")
        if fmri.shape != (FMRI_VOXEL_COUNT,):
            raise ValueError(f"Unexpected fMRI shape for subject {trial['subject_id']}, file {filename}: {fmri.shape}.")
        eeg_segments.append(eeg)
        fmri_frames.append(fmri)

    # hooks see continuous trial EEG and stacked fMRI, with explicit output timing checks
    eeg = preprocess_eeg(np.concatenate(eeg_segments, axis = -1), preprocessing_functions = EEG_PREPROCESSING_FUNCTIONS)
    fmri = preprocess_fmri(np.stack(fmri_frames, axis = 0), preprocessing_functions = FMRI_PREPROCESSING_FUNCTIONS)
    expected_samples = integer_ratio(FMRI_FRAMES_PER_TRIAL * FMRI_TR_SECONDS * EEG_OUTPUT_SAMPLING_RATE, 1, "Output EEG samples")
    if eeg.shape != (EEG_CHANNEL_COUNT, expected_samples):
        raise ValueError("EEG hooks changed the trial duration or disagreed with EEG_OUTPUT_SAMPLING_RATE.")
    return eeg, fmri


def generation_metadata(counts: dict[str, int], report: dict[str, Any]) -> dict[str, Any]:
    # retain configuration and assumptions so the inferred mapping can be audited later
    return {
        "status": "incomplete", "dataset_path": str(Path(DATASET_PATH).resolve()),
        "frames_per_trial": FMRI_FRAMES_PER_TRIAL, "fmri_tr_seconds": FMRI_TR_SECONDS,
        "trial_duration_seconds": FMRI_FRAMES_PER_TRIAL * FMRI_TR_SECONDS,
        "eeg_sampling_rate": EEG_OUTPUT_SAMPLING_RATE, "video_sampling_rate": VIDEO_SAMPLING_RATE,
        "video_layout": "THWC RGB uint8", "eeg_layout": "channels, samples",
        "video_storage": "disk", "video_codec": "libx264", "video_crf": VIDEO_CRF,
        "video_preset": VIDEO_PRESET,
        "video_filename": "{first_clip_id:06d}_{first_frame:04d}_{last_clip_id:06d}_{last_frame:04d}.mp4; one-based inclusive frame numbers",
        "clips_path": str((Path(DATAFOLDER_PATH) / "clips").resolve()),
        "fmri_layout": "frames, voxels", "video_timing": "endpoint-inclusive; nearest source frame; exact trial timestamps stored",
        "temporal_shift_seconds": 0, "counts": counts, "report": report,
        "mapping_verified": False,
        "mapping_assumptions": "Subject-local neural indices in behavioural run order; global clips in VIDEO_EPISODE_ORDER. Neither numbering scheme is established by the public loader.",
        "video_episode_order": VIDEO_EPISODE_ORDER, "neural_run_starts": NEURAL_RUN_STARTS,
        "video_episode_starts": VIDEO_EPISODE_STARTS,
        "neural_first_index": NEURAL_FIRST_INDEX, "video_first_index": VIDEO_FIRST_INDEX,
        "train_runs": TRAIN_RUNS, "val_runs": VAL_RUNS, "test_runs": TEST_RUNS,
        "eeg_hooks": [repr(function) for function in EEG_PREPROCESSING_FUNCTIONS or []],
        "fmri_hooks": [repr(function) for function in FMRI_PREPROCESSING_FUNCTIONS or []],
        "references": ["https://arxiv.org/html/2503.06940v2",
                       "https://github.com/yanweifu-sii/CineBrain/blob/main/data_video.py",
                       "https://huggingface.co/datasets/Fudan-fMRI/CineBrain"],
    }


# Entry point
# run with `python -m dataset.CineBrain.gen_database`
def main() -> None:
    # check the setup before creating any database files
    counts = validate_setup()
    log(f"Preparing {FMRI_FRAMES_PER_TRIAL}-frame trials ({FMRI_FRAMES_PER_TRIAL * FMRI_TR_SECONDS:g} s); EEG and fMRI hooks are opt-in.")
    log("Neural and video numbering are inferred, not verified by upstream annotations. Review the setup overrides against the remote export.", "WARN")
    # this 'plan' plans every trail to be extracted (subject has some runs on some episodes)
    plan, report = make_plan(counts)
    # use the explicit output root for the report and generated video clips
    datafolder = Path(DATAFOLDER_PATH)
    datafolder.mkdir(parents = True, exist_ok = True)
    # output path (lmdb), the parent dir should exist
    output = Path(DATABASE_PATH)
    output.parent.mkdir(parents = True, exist_ok = True)
    # report path (JSON)
    report_path = datafolder / "generation_report.json"
    report_path.write_text(json.dumps(report, indent = 2), encoding = "utf-8")
    log(f"Discard and mapping report saved to '{report_path}'.")
    confirm_discard(report)

    # check video dependency library
    try:
        import av
    except ImportError as exc:
        raise ImportError("PyAV missing. Install with 'pip install av'.") from exc
    # set up the H.264 encoder for MP4 output; PyAV will raise if unavailable
    av.codec.Codec("libx264", "w")

    # create dir for generated clips (to serve our custom trails)
    clips_path = datafolder / "clips"
    clips_path.mkdir(parents = True, exist_ok = True)
    log(f"Saving H.264 MP4 clips to '{clips_path}' (CRF {VIDEO_CRF}); filenames use one-based, inclusive source frame numbers.")
    
    # generate metadata for the database
    metadata = generation_metadata(counts, report)

    cache: OrderedDict[int, np.ndarray] = OrderedDict()
    saved_clips: dict[str, dict[str, Any]] = {}

    # create the database and write all trials and their metadata
    with Database(str(output), mode = "c", map_size = DATABASE_MAP_SIZE) as database:
        database.write_meta_name(DATABASE_NAME)
        database.write_db_entry("meta.generation", metadata, overwrite = False) # new entry
        try:
            for position, trial in enumerate(plan, start = 1):
                # get resources
                eeg, fmri = load_trial_neural(trial, counts)
                video, video_times = save_trial_video(trial, counts, clips_path, cache, saved_clips)
                
                # insert trail
                trial_idx = database.insert_trial(
                    subject_id = trial["subject_id"], phase = trial["phase"], state = TRIAL_STATE,
                    resources = {"eeg": eeg, "fmri": fmri, "video": video, "video_times": video_times},
                )

                # attach provenance to the existing trial record without altering its resources
                # NOTE: adds attributes
                trial_metadata = database.get_db_attribute(f"meta.trial_{trial_idx}")
                trial_metadata["cinebrain"] = dict(trial)
                trial_metadata["cinebrain"].update({
                    "duration_seconds": FMRI_FRAMES_PER_TRIAL * FMRI_TR_SECONDS,
                    "episode_start_seconds": trial["frame_offset"] * FMRI_TR_SECONDS,
                    "eeg_sampling_rate": EEG_OUTPUT_SAMPLING_RATE,
                    "fmri_tr_seconds": FMRI_TR_SECONDS,
                })
                database.write_db_entry(f"meta.trial_{trial_idx}", trial_metadata, overwrite = True) # previously existing entry
                if position % PROGRESS_EVERY_TRIALS == 0 or position == len(plan):
                    log(f"Stored {position}/{len(plan)} trials and {len(saved_clips)} unique MP4 clips.")

            # mark success only after every planned trial and its metadata has been written
            metadata["status"] = "complete"
            database.write_db_entry("meta.generation", metadata, overwrite = True) # previously existing entry
            database.print_brief()
        except BaseException:
            log(f"Generation stopped. '{output}' is incomplete; choose a new output path for a fresh run. No additional corrupt data were silently discarded.", "ERROR")
            raise
    log(f"Created '{output}' (lmdb PATH) with {len(plan)} trials referencing {len(saved_clips)} MP4 clips in '{clips_path}'.", "OK")


if __name__ == "__main__":
    main()
