# file: tensorboard_extractor.py
# exports tensorboard scalar metrics
# standalone script / run with `python results/tensorboard_extractor.py`

from __future__ import annotations

# [SETUP] ===============================
TARGET_LOGDIR = '../<logdir>'
OUTPUT_FILE = './<name>.pdf'
RASTERISE_CURVES = True           # Rasterise data only; (text and axes remain vector graphics)
RASTER_DPI = 200                  # Resolution of rasterised curves in the PDF
FIGURE_SIZE = (8.5, 5.0)          # Width and height in inches
LINE_WIDTH = 1.2
# =======================================

# import libraries
import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util


def log(message: str, level: str = 'INFO') -> None:
    colour = {'INFO': 34, 'WARN': 33, 'OK': 32, 'ERROR': 31}[level]
    print(f'\033[1;{colour}m[TENSORBOARD] {level}\033[0m: {message}', flush = True)


def extract_and_export_pdf(log_dir: str | Path, output_pdf_path: str | Path,
                           rasterise: bool = RASTERISE_CURVES,
                           dpi: int = RASTER_DPI) -> int:
    """
    Export one page per run and scalar tag; return the number of pages written
    """
    directory = Path(log_dir).expanduser().resolve()
    destination = Path(output_pdf_path).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f'TensorBoard log directory does not exist: {directory}')
    if destination.suffix.lower() != '.pdf' or dpi <= 0:
        raise ValueError('Choose a .pdf output filename and a positive raster DPI.')

    # Group event files by directory, so resumed runs do not create duplicate reports.
    event_files = sorted(path for path in directory.rglob('*events.out.tfevents*') if path.is_file())
    run_dirs = sorted({path.parent for path in event_files})
    if not run_dirs:
        raise FileNotFoundError(f'No TensorBoard event files found under {directory}')
    log(f'Found {len(event_files)} event files in {len(run_dirs)} run directories.')
    log(f'Curve rendering: {"rasterised" if rasterise else "vector"}; raster resolution: {dpi} DPI.')

    # Replace the destination only after a non-empty report has been completed.
    destination.parent.mkdir(parents = True, exist_ok = True)
    with NamedTemporaryFile(suffix = '.pdf', dir = destination.parent, delete = False) as handle:
        temporary = Path(handle.name)
    pages = 0
    try:
        with PdfPages(temporary, metadata = {'Title': 'TensorBoard scalar metrics',
                                            'Subject': str(directory)}) as pdf:
            for index, run_dir in enumerate(run_dirs, start = 1):
                relative = run_dir.relative_to(directory)
                run_name = str(relative) if str(relative) != '.' else directory.name
                log(f'Run {index}/{len(run_dirs)} ({100 * index / len(run_dirs):.1f}%): {run_name}')

                # Disable reservoir sampling: retain all points and honour TensorBoard restart markers.
                accumulator = event_accumulator.EventAccumulator(
                    str(run_dir), size_guidance = {'scalars': 0, 'tensors': 0},
                    purge_orphaned_data = True,
                )
                accumulator.Reload()
                tags = accumulator.Tags()
                scalar_tags = set(tags.get('scalars', []))
                # TensorFlow also stores scalar summaries as tensors tagged by the scalars plugin.
                tensor_tags: set[str] = set()
                for tag in tags.get('tensors', []):
                    if accumulator.SummaryMetadata(tag).plugin_data.plugin_name == 'scalars':
                        tensor_tags.add(tag)
                if not scalar_tags and not tensor_tags:
                    log(f'No scalar metrics in {run_name}; skipping.', 'WARN')
                    continue

                for tag in sorted(scalar_tags | tensor_tags):
                    points: list[tuple[int, float, float]] = []
                    if tag in scalar_tags:
                        points.extend((event.step, event.wall_time, float(event.value))
                                      for event in accumulator.Scalars(tag))
                    if tag in tensor_tags:
                        for event in accumulator.Tensors(tag):
                            value = tensor_util.make_ndarray(event.tensor_proto)
                            if value.size == 1 and np.issubdtype(value.dtype, np.number) and not np.iscomplexobj(value):
                                points.append((event.step, event.wall_time, float(value.item())))
                    if not points:
                        log(f'No numeric scalar points for {run_name}: {tag}; skipping.', 'WARN')
                        continue

                    # Keep the latest observation for duplicate steps, then plot in step order.
                    latest: dict[int, float] = {}
                    for step, wall_time, value in sorted(points, key = lambda point: (point[0], point[1])):
                        latest[step] = value
                    steps = np.asarray(list(latest))
                    values = np.asarray(list(latest.values()), dtype = np.float64)
                    finite = np.isfinite(values)
                    if not finite.any():
                        log(f'All values are non-finite for {run_name}: {tag}; skipping.', 'WARN')
                        continue
                    if not finite.all():
                        log(f'{run_name}: {tag} contains {int((~finite).sum())} non-finite values; showing gaps.', 'WARN')
                        values[~finite] = np.nan

                    # Rasterise only the dense curve; labels, grid and axes stay sharp at any zoom.
                    fig, ax = plt.subplots(figsize = FIGURE_SIZE)
                    try:
                        ax.plot(steps, values, color = '#1f77b4', linewidth = LINE_WIDTH,
                                rasterized = rasterise)
                        ax.set_title(f'Run: {run_name}\nMetric: {tag.replace("/", " / ")}',
                                     fontsize = 12, fontweight = 'bold', pad = 12, wrap = True)
                        ax.set_xlabel('Step')
                        ax.set_ylabel('Value')
                        ax.grid(True, linestyle = '--', alpha = 0.4)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        fig.tight_layout()
                        pdf.savefig(fig, dpi = dpi)
                        pages += 1
                    finally:
                        plt.close(fig)
                    log(f'  Page {pages}: {tag} ({len(steps):,} steps).')
                del accumulator
        if pages == 0:
            raise ValueError('No plottable scalar metrics found; the output PDF was not replaced.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok = True)
    log(f'Saved {pages} pages to {destination}', level = 'OK')
    return pages


if __name__ == '__main__':
    # [OPTIONAL] Argparse overrides
    parser = argparse.ArgumentParser(description = 'Tensorboard logs (scalars) to PDF')
    parser.add_argument('log_dir', nargs = '?', default = TARGET_LOGDIR)
    parser.add_argument('output_pdf', nargs = '?', default = OUTPUT_FILE)
    parser.add_argument('--rasterise', action = argparse.BooleanOptionalAction,
                        default = RASTERISE_CURVES, help = 'Rasterise curves while retaining vector text and axes.')
    parser.add_argument('--dpi', type = int, default = RASTER_DPI, help = 'Raster resolution in dots per inch.')
    args = parser.parse_args()
    try:
        extract_and_export_pdf(args.log_dir, args.output_pdf, rasterise = args.rasterise, dpi = args.dpi)
    except (OSError, ValueError) as error:
        log(str(error), level = 'ERROR')
        parser.exit(1)
