# file: preprocessing_hook.py
# shared (maybe used in EEG/fMRI) function hooks for the supplied CineBrain arrays

from __future__ import annotations
from collections.abc import Callable, Sequence
import numpy as np

PreprocessingFunctions = Sequence[Callable[[np.ndarray], np.ndarray]] | None


def apply_preprocessing(data: np.ndarray, preprocessing_functions: PreprocessingFunctions = None) -> np.ndarray:
    """
    Apply ordered array-to-array functions; None preserves the supplied values.

    Use functools.partial to bind additional parameters to a function. Hooks
    receive a copy, so in-place operations do not alter the source array.
    """
    # validate numerical data without changing its shape, dtype or scale
    if not isinstance(data, np.ndarray) or data.dtype.kind not in "iuf":
        raise ValueError("\033[1;31m[ERROR]\033[0m: Expected a real numerical NumPy array.")
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("\033[1;31m[ERROR]\033[0m: Data are empty or contain non-finite values.")

    # leave already processed data untouched unless hooks are explicitly supplied
    if preprocessing_functions is None or len(preprocessing_functions) == 0:
        return data
    result = data.copy()
    for function in preprocessing_functions:
        result = function(result)
        if not isinstance(result, np.ndarray) or result.dtype.kind not in "iuf":
            raise ValueError("\033[1;31m[ERROR]\033[0m: Each preprocessing hook must return a real numerical NumPy array.")
        if result.size == 0 or not np.isfinite(result).all():
            raise ValueError("\033[1;31m[ERROR]\033[0m: A preprocessing hook returned empty or non-finite data.")
    return result


def normalise(data: np.ndarray, axis: int = -1, epsilon: float = 1e-8) -> np.ndarray:
    """
    Optionally z-score along an explicit axis; constant signals become zero.

    EEG time is axis -1; stacked fMRI time is axis 0. This function is never
    enabled by default. Fit any dataset-wide statistics on training data only.
    """
    # use floating point arithmetic and protect constant channels from division by zero
    if epsilon <= 0:
        raise ValueError("\033[1;31m[ERROR]\033[0m: epsilon must be positive.")
    values = data.astype(np.float64, copy = False)
    return (values - values.mean(axis = axis, keepdims = True)) / np.maximum(
        values.std(axis = axis, keepdims = True), epsilon,
    )
