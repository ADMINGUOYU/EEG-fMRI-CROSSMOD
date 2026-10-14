# file: dataset.py
# This module provides a general dataset wrapper around the standardised
# database implementation in :mod:`dataset.database`.
# The base class is intentionally lightweight and extendable:
# - subclasses may override trial sampling,
# - subclasses may override sample processing,
# - the default collation function works with dictionaries of NumPy arrays,
#   tensors, numbers, strings, and nested structures.

# import libraries
from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from dataset.database import Database


# class for dataset
class Dataset(torch.utils.data.Dataset):

    # init function
    def __init__(
        self,
        db_path: str,
        trial_indices: Sequence[int] | None = None,
        subject_id: int | None = None,
        phase: str | None = None,
        state: str | None = None,
        modalities: Sequence[str] | None = None,
        return_trial_idx: bool = True,
        return_meta: bool = True,
        preload: bool = False,
        transform: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        """
        Initialises a general dataset wrapper around the database backend.

        Arguments:
            db_path           : str | path to the database directory.
            trial_indices     : Sequence[int] | explicit trial selection.
            subject_id        : int | optional subject filter.
            phase             : str | optional phase filter.
            state             : str | optional state filter.
            modalities        : Sequence[str] | optional modality selection.
            return_trial_idx  : bool | include the selected trial index in each sample.
            return_meta       : bool | include trial metadata in each sample.
            preload           : bool | cache decoded samples in memory.
            transform         : callable | optional runtime transform for each sample.
        """

        # save the configuration first so helper methods can use it safely.
        self.db_path = os.fspath(db_path)
        self.return_trial_idx = bool(return_trial_idx)
        self.return_meta = bool(return_meta)
        self.preload = bool(preload)
        self.transform = transform

        # keep the selection filters for printing and for later refreshes.
        self.subject_id = subject_id
        self.phase = phase
        self.state = state
        self.modalities = tuple(modalities) if modalities is not None else None

        # open the standardised database backend. (always read-only)
        self.db = Database(self.db_path, mode = 'r')

        # build the trial list once so sampling stays deterministic unless refreshed.
        self.trial_indices = self._resolve_trial_indices(
            trial_indices = trial_indices,
            subject_id = subject_id,
            phase = phase,
            state = state,
        )

        # initialise an optional cache for preloaded samples.
        self._cache: dict[int, dict[str, Any]] = {}

        # preload samples if requested by the caller.
        if self.preload:
            self._preload_samples()

        # print a concise briefing in the same spirit as the database class.
        self.print_brief()

    # [ACCESSOR] fetch the number of selected trials
    def __len__(self) -> int:
        return len(self.trial_indices)

    # [ACCESSOR] retrieve one sample by dataset index
    def __getitem__(self, index: int) -> Any:
        """
        Returns a single processed sample.

        Subclasses may override :meth:`process_sample` to customise runtime
        preprocessing without changing the database access layer.
        """

        # resolve the public dataset index into the underlying trial index.
        trial_idx = self._resolve_trial_idx(index)

        # reuse a cached sample when preloading is enabled.
        if trial_idx in self._cache:
            sample = self._clone_sample(self._cache[trial_idx])
        else:
            sample = self.load_trial(trial_idx)
            if self.preload:
                self._cache[trial_idx] = self._clone_sample(sample)

        # give subclasses a single hook for arbitrary runtime processing.
        sample = self.process_sample(sample, trial_idx = trial_idx)

        # apply an optional caller-supplied transform after subclass processing.
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # [ACCESSOR] collate function for PyTorch DataLoader
    @staticmethod
    def collate_fn(batch: list[Any]) -> Any:
        """
        Collates a batch of samples into a single batched structure.

        This default implementation uses PyTorch's standard collate behaviour,
        which works well for nested dictionaries, NumPy arrays, tensors, numbers,
        and strings.
        """

        # return an empty batch cleanly instead of raising a less helpful error.
        if len(batch) == 0:
            return batch

        return default_collate(batch)

    # [ACCESSOR] load one trial from the database
    def load_trial(self, trial_idx: int) -> dict[str, Any]:
        """
        Loads a single trial from the database and returns a sample dictionary.

        The returned sample keeps the database payload general so subclasses can
        reshape, augment, or reformat it as needed.
        """

        # fetch the trial metadata first so we know which modalities are available.
        trial_meta = self.db.get_db_attribute(f'meta.trial_{int(trial_idx)}')

        # select the required modalities or fall back to all stored modalities.
        modality_names = self._resolve_modalities(trial_meta)

        # decode the requested modalities from the database.
        data = {
            modality_name: self.db.read_trial_modality(int(trial_idx), modality_name)
            for modality_name in modality_names
        }

        # build the standard sample payload.
        sample: dict[str, Any] = {
            'data': data,
        }

        # add useful metadata when requested.
        if self.return_trial_idx:
            sample['trial_idx'] = int(trial_idx)
        if self.return_meta:
            sample['meta'] = trial_meta

        return sample

    # [ACCESSOR] give subclasses a single extension point for processing
    def process_sample(self, sample: dict[str, Any], trial_idx: int | None = None) -> Any:
        """
        Applies runtime processing to a sample.

        The default implementation is a no-op and simply returns the sample.
        Subclasses can override this method to normalise, crop, pad, filter, or
        reformat samples without touching the database logic.
        """

        return sample

    # [ACCESSOR] refresh the trial selection
    def refresh(self) -> None:
        """
        Rebuilds the trial list from the current filters and clears any cache.
        """

        # rebuild the selectable trial indices from the stored configuration.
        self.trial_indices = self._resolve_trial_indices(
            trial_indices = None,
            subject_id = self.subject_id,
            phase = self.phase,
            state = self.state,
        )

        # clear the cache because the visible dataset may have changed.
        self._cache.clear()

        # reload the cache if preloading is enabled.
        if self.preload:
            self._preload_samples()

    # [ACCESSOR] print a concise briefing about this dataset
    def print_brief(self) -> None:
        """
        Prints a short human-readable summary of the dataset configuration.
        """

        # gather the database name if it is available.
        try:
            db_name = self.db.get_db_attribute('meta.name')
        except Exception:
            db_name = ''

        # collect a few simple statistics for the selected subset.
        selected_meta = self._collect_selected_meta()
        subject_ids = {int(item.get('subject_id')) for item in selected_meta if item.get('subject_id') is not None}
        phase_counts = Counter(str(item.get('phase')) for item in selected_meta if item.get('phase') is not None)
        state_counts = Counter(str(item.get('state')) for item in selected_meta if item.get('state') is not None)

        # print the briefing using the same overall style as database.py.
        print(f"\033[1;34m[Dataset] INFO\033[0m: (print_brief) Dataset at '{self.db_path}'")
        print(f"  Name: {db_name}")
        print(f"  Trial Count (selected): {len(self.trial_indices)}")
        print(f"  Subject Count (selected): {len(subject_ids)}")

        # print the active filters for clarity.
        print("  Selection Filters:")
        print(f"    Subject ID: {self.subject_id if self.subject_id is not None else 'all'}")
        print(f"    Phase: {self.phase if self.phase is not None else 'all'}")
        print(f"    State: {self.state if self.state is not None else 'all'}")
        print(f"    Modalities: {', '.join(self.modalities) if self.modalities is not None else 'all'}")
        print(f"    Preload: {'enabled' if self.preload else 'disabled'}")

        # print split statistics for the selected subset.
        print(f"  Phase/Split Count: {len(phase_counts)}")
        for phase, count in phase_counts.items():
            ratio = count / max(len(self.trial_indices), 1)
            print(f"    Phase '{phase}': {count} trials ({ratio:.2%})")

        # print experimental state statistics for the selected subset.
        print(f"  State/Condition Count: {len(state_counts)}")
        for state, count in state_counts.items():
            ratio = count / max(len(self.trial_indices), 1)
            print(f"    State '{state}': {count} trials ({ratio:.2%})")

    # [Mutator] close the database handle
    def close(self) -> None:
        """
        Synchronises and closes the underlying database connection.
        """

        # close the database if it is still alive.
        if hasattr(self, 'db') and self.db is not None:
            self.db.close()
            self.db = None

        print(f"\033[1;34m[Dataset] INFO\033[0m: (close) Closed Dataset at '{self.db_path}'")

    # TO WORK WITH python 'with' block
    def __enter__(self):
        return self

    # TO WORK WITH python 'with' block
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # [ACCESSOR] textual representation for interactive use
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(db_path = {self.db_path!r}, "
            f"trials = {len(self.trial_indices)}, preload = {self.preload})"
        )

    # [INTERNAL] resolve dataset indices to underlying trial indices
    def _resolve_trial_indices(
        self,
        *,
        trial_indices: Sequence[int] | None,
        subject_id: int | None,
        phase: str | None,
        state: str | None,
    ) -> list[int]:
        """
        Resolves the selected trial indices from the database and user filters.
        """

        # start from database-level filtering when any filter is requested.
        if trial_indices is None:
            selected = set(self.db.filter_trials(subject_id = subject_id, phase = phase, state = state))
        else:
            selected = {int(index) for index in trial_indices}

            # if database filters are provided as well, intersect them with the explicit list.
            if subject_id is not None or phase is not None or state is not None:
                filtered = set(self.db.filter_trials(subject_id = subject_id, phase = phase, state = state))
                selected = selected.intersection(filtered)

        return sorted(selected)

    # [INTERNAL] convert a dataset index into a trial index
    def _resolve_trial_idx(self, index: int) -> int:
        """
        Converts a public dataset index into the corresponding trial index.
        """

        # support negative indexing in the usual Python manner.
        if index < 0:
            index += len(self.trial_indices)

        # validate the final index range explicitly.
        if index < 0 or index >= len(self.trial_indices):
            raise IndexError(f"\033[1;31m[Dataset] ERROR\033[0m: (_resolve_trial_idx) Index '{index}' is out of range for dataset of length {len(self.trial_indices)}.")

        return int(self.trial_indices[index])

    # [INTERNAL] work out which modalities to read for one trial
    def _resolve_modalities(self, trial_meta: dict[str, Any]) -> list[str]:
        """
        Returns the modality names that should be loaded for a trial.
        """

        available = list(trial_meta.get('data', {}).keys())

        # if the caller did not specify modalities, load everything available.
        if self.modalities is None:
            return available

        # otherwise keep only the modalities that are actually present.
        missing = [modality_name for modality_name in self.modalities if modality_name not in available]
        if len(missing) > 0:
            raise KeyError(
                f"\033[1;31m[Dataset] ERROR\033[0m: (_resolve_modalities) Requested modality or modalities {missing} "
                f"are not available in the selected trial."
            )

        return [str(modality_name) for modality_name in self.modalities]

    # [INTERNAL] preload all selected samples into memory
    def _preload_samples(self) -> None:
        """
        Decodes every selected sample once and stores it in the cache.
        """

        # iterate over the selected trial indices in a stable order.
        for trial_idx in self.trial_indices:
            self._cache[int(trial_idx)] = self.load_trial(int(trial_idx))

    # [INTERNAL] collect metadata for all selected trials
    def _collect_selected_meta(self) -> list[dict[str, Any]]:
        """
        Returns the metadata dictionaries for all selected trials.
        """

        selected_meta: list[dict[str, Any]] = []

        # read each trial entry individually so we can summarise the visible subset.
        for trial_idx in self.trial_indices:
            try:
                trial_meta = self.db.get_db_attribute(f'meta.trial_{int(trial_idx)}')
            except Exception:
                continue

            if isinstance(trial_meta, dict):
                selected_meta.append(trial_meta)

        return selected_meta

    # [INTERNAL] clone a sample safely for cache reuse
    def _clone_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Makes a shallow copy of the sample and copies the top-level data dict.
        """

        cloned = dict(sample)
        if 'data' in cloned and isinstance(cloned['data'], dict):
            cloned['data'] = dict(cloned['data'])
        return cloned


# Entry point (testing)
# run with `python -m dataset.dataset`
if __name__ == "__main__":

    # import libraries for testing
    import shutil
    import tempfile

    # Create temp dir
    tmp = tempfile.mkdtemp()
    db_dir = os.path.join(tmp, "db")
    print("\033[1;34m[Dataset] INFO\033[0m: Starting dataset smoke test")
    print(f"\033[1;34m[Dataset] INFO\033[0m: Temp Database Path: {db_dir}\n")

    try:
        # instantiate a tiny database for a quick smoke test.
        db = Database(db_dir, mode = 'c', map_size = int(10 * 1024 * 1024))
        db.write_meta_name("TestDB")

        # build a minimal trial with one in-memory and one disk-backed modality.
        dummy_eeg = np.random.rand(8, 4).astype(np.float32)
        dummy_signal_path = os.path.join(tmp, "dummy_signal.npy")
        np.save(dummy_signal_path, np.random.rand(4, 4).astype(np.float32))

        resources = {
            'eeg': dummy_eeg,
            'signal': {
                'path': dummy_signal_path,
                'type': 'npy',
                'shape': (4, 4),
                'dtype': 'float32'
            }
        }
        trial_idx = db.insert_trial(subject_id = 1, phase = 'train', state = 'resting', resources = resources)
        db.close()

        # reopen the database through the dataset wrapper.
        dataset = Dataset(db_dir)
        assert len(dataset) == 1, "\033[1;31m[Dataset] ERROR\033[0m: Expected one trial in the dataset."

        # fetch the sample and verify the trial index and data are present.
        sample = dataset[0]
        assert sample['trial_idx'] == trial_idx, "\033[1;31m[Dataset] ERROR\033[0m: Trial index mismatch."
        assert 'data' in sample and 'eeg' in sample['data'], "\033[1;31m[Dataset] ERROR\033[0m: Missing EEG data."

        # print a short summary and close the wrapper.
        dataset.print_brief()
        dataset.close()

    finally:
        # Cleanup test directories safely.
        shutil.rmtree(tmp, ignore_errors = True)

    print("\033[1;32m[Dataset] OK\033[0m: Test completed successfully.")
