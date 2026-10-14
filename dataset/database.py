# file: database.py
# WHAT: this file implements a databases using lmdb
#       the database is used to store metadata & data for the brain dataset.
#       the database provides standard APIs for the dataset to access
#       data and metadata.
#       For a specific dataset (i.e. CineBrain), you have to implement code
#       to stuff this database with correct data & meta layout (so our downstream
#       code works as expected).
# DATABASE LAYOUT: (2 databases, data & meta)
# A) 'meta': stores metadata for the dataset (i.e. data mapping)
#     - Reserved keys:
#       - 'meta': basic attributes (dictionary of)
#              'name' : name of the dataset
#              'subject_dict' : {
#                                   '<subject_id>(int)' : {'trial_n' : (number of trials), ... (extensible) }
#                                                           # equals to the entries corresponding to this subject
#                               }
#              'trial_n' : total number of trials in the dataset (should match the sum of above)
#
#     - Meta keys:
#       - 'trial_<trial_idx:int>': mapping information for a single trial
#              'subject_id' : int
#              'phase'      : str ('train' | 'val' | 'test')
#              'state'      : str ('resting' | 'task' | 'motor_imagery' etc.)
#              'data'       : dict of {
#                               '<modality_name>(i.e. eeg, fmri, etc.)' : {
#                                   'storage' : str,     # 'lmdb' for database blocks, 'disk' for external files
#                                   # --- If 'storage' == 'lmdb' ---
#                                   'key' : bytes,       # The physical database key in 'data' sub-db
#                                   'offset' : int,      # Byte offset start in raw binary block
#                                   # --- If 'storage' == 'disk' ---
#                                   'path' : str,        # Absolute or relative file path on disk
#                                   'type' : str,        # File format/extension (e.g., 'mp4', 'nii')
#                                   'offset_index' : int,# Frame/sample offset index (e.g. video starting frame)
#                                   # --- Shared parameters ---
#                                   'shape' : tuple,     # Array dimensions (e.g., (800, 64) for EEG; (frames, H, W, C) for video)
#                                   'dtype' : str        # Array data type (e.g., 'float32', 'int16')
#                               }
#                             }
#
#     - Query/Inverted Index keys:
#       - 'idx_subjects': dict of {<subject_id>(int): [int, int, ...]} where each list is a list of trial indices belonging to a subject
#       - 'idx_phase'   : dict of {<phase_name>(str): [int, int, ...]} where each list is a list of trial indices belonging to a phase split
#       - 'idx_state'   : dict of {<state_name>(str): [int, int, ...]} where each list is a list of trial indices belonging to an experimental condition
#
# B) 'data': stores the raw high-dimensional multi-modal binary buffers
#     - Data keys (fast index-mapped direct pointers):
#       - '<data_key>'   : raw binary buffer bytes (i.e via ndarray.tobytes())


# import libraries
from __future__ import annotations
import os
import lmdb
import pickle
import hashlib
import numpy as np
from typing import Any


# Database class
class Database:

    def __init__(
            self,
            db_path: str,
            mode: str,
            map_size: int = int(1e12)
        ) -> None:
        """
        initialises the LMDB Environment (with max_dbs = 2)

        Arguments:
            db_path : str : path to the database directory (DIR!!! NOT file)
            mode    : str : 'r' for read-only, 'a' for append(no delete), 'c' for create (new)
                            if 'a' we will let you confirm the file
            map_size: int : maximum size of the database in bytes (default: 1TB)
        """

        # check environment
        if mode not in ['r', 'a', 'c']:
            raise ValueError(f"\033[1;31m[Database] ERROR\033[0m: (__init__) Invalid mode '{mode}'. Must be one of ['r', 'a', 'c'].")
        # (path existence here for customised warning)
        if os.path.exists(db_path):
            if mode == 'c':
                raise FileExistsError(f"\033[1;31m[Database] ERROR\033[0m: (__init__) Database path '{db_path}' already exists. Cannot create a new database in an existing directory.")
            elif mode == 'a':
                print(f"\033[1;33m[Database] WARN\033[0m: (__init__) Database path '{db_path}' exists. Opening in append mode.")
                # ask if to continue (for safety)
                user_input = input(f"\033[1;33m[Database] WARN\033[0m: (__init__) Continue? (y/n): ")
                if user_input.lower() != 'y':
                    raise RuntimeError(f"\033[1;34m[Database] INFO\033[0m: (__init__) User aborted operation. Exiting.")
        else:
            if mode in ['r', 'a']:
                raise FileNotFoundError(f"\033[1;31m[Database] ERROR\033[0m: (__init__) Database path '{db_path}' does not exist. Cannot open in read-only or read-write mode.")
            elif mode == 'c':
                print(f"\033[1;34m[Database] INFO\033[0m: (__init__) Creating new database at '{db_path}'.")

        # save a copy of current mode
        self.mode = mode

        # save a copy of the database path
        self.db_path = db_path

        # initialise LMDB environment
        self.env = lmdb.open(
            db_path,
            map_size = map_size,
            max_dbs = 2,
            readonly = (mode == 'r'),
            create = (mode == 'c'),
            lock = (mode != 'r')
        )

        # open sub-databases
        self.meta_db = self.env.open_db(b'meta')
        self.data_db = self.env.open_db(b'data')

        # if create, set up empty meta structure
        if mode == 'c':
            with self.env.begin(write = True) as txn:
                # initialise 'meta' dictionary
                meta_dict = {
                    'name': '',
                    'subject_dict': {},
                    'trial_n': 0,
                    'idx_subjects': {},
                    'idx_phase': {},
                    'idx_state': {}
                }
                # write to 'meta' sub-database
                txn.put(b'meta', pickle.dumps(meta_dict), db = self.meta_db)

    # [Mutator] update meta-data 'name'
    def write_meta_name(self, name: str):
        """
        In the 'meta' sub-database, sets 'meta'['name'] = name.
        Our database will count the 'subject_dict', 'trial_n' and
        the inverted index lists automatically when you write trial entries.
        """
        meta_dict = self._read_meta_dict()
        meta_dict['name'] = str(name)
        self._write_meta_dict(meta_dict)

    # [Mutator] insert a whole trial in one call
    def insert_trial(self, subject_id: int, phase: str, state: str, resources: dict, trial_idx: int = None) -> int:
        """
        Inserts a single trial and all of its modalities/resources in one convenient call.

        NOTE: resource (dict format)
        {
            'storage' : str,     # 'lmdb' for database blocks, 'disk' for external files
            # --- If 'storage' == 'lmdb' ---
            'value'  : Any arr,  # Any array liked data
            'key'    : bytes,    # !!! only if you have it, The physical database key in 'data' sub-db (normally, you don't)
            'offset' : int,      # !!! Byte offset start in raw binary block
            # --- If 'storage' == 'disk' ---
            'path' : str,        # Absolute or relative file path on disk
            'type' : str,        # File format/extension (e.g., 'mp4', 'nii')
            'offset_index' : int,# Frame/sample offset index (e.g. video starting frame)
            # --- Shared parameters ---
            'shape' : tuple,     # Array dimensions (e.g., (800, 64) for EEG; (frames, H, W, C) for video)
            'dtype' : str        # Array data type (e.g., 'float32', 'int16')
        }

        Example:
            resources = {
                'eeg': eeg_array,
                'fmri': '/path/to/subject_001.nii',
                'video': {'path': '/path/to/video.mp4', 'type': 'mp4'},
                '<else>': {'value' : some array, 'shape': (frames, H, W, C), 'dtype': 'float32', 'offset': 0}
            }
            trial_idx, resource_key = db.insert_trial(subject_id = 1, phase = 'train', state = 'resting', resources = resources)
        """

        # read the current meta dictionary for the index update
        meta_dict = self._read_meta_dict()

        # determine the trial index: if not provided, generate a new one
        if trial_idx is None:
            trial_idx = self._get_new_trial_index()

        # refuse to overwrite an existing trial entry
        trial_key = f"trial_{trial_idx}"
        with self.env.begin(write = False) as txn:
            if txn.get(trial_key.encode('utf-8'), db = self.meta_db) is not None:
                raise FileExistsError(f"\033[1;31m[Database] ERROR\033[0m: (insert_trial) Trial '{trial_key}' already exists.")

        # build the modality mapping for this trial
        # resources
        resource_dict = {}
        for modality_name, resource in resources.items():
            if modality_name is None or modality_name == '':
                raise ValueError(f"\033[1;31m[Database] ERROR\033[0m: (insert_trial) Modality/resource name cannot be empty.")
            resource_dict[modality_name] = self._make_resource(resource, modality_name = modality_name)

        # create the trial entry
        trial_entry = {
            'subject_id': int(subject_id),
            'phase': str(phase),
            'state': str(state),
            'data': resource_dict
        }

        # write the trial and update the inverted indexes together
        with self.env.begin(write = True) as txn:
            txn.put(trial_key.encode('utf-8'), pickle.dumps(trial_entry), db = self.meta_db)

            subject_entry = meta_dict.setdefault('subject_dict', {}).setdefault(int(subject_id), {'trial_n': 0})
            subject_entry['trial_n'] = int(subject_entry.get('trial_n', 0)) + 1
            subject_entry.setdefault('trials', []).append(trial_idx)

            idx_subjects = meta_dict.setdefault('idx_subjects', {})
            idx_subjects.setdefault(int(subject_id), []).append(trial_idx)

            idx_phase = meta_dict.setdefault('idx_phase', {})
            idx_phase.setdefault(str(phase), []).append(trial_idx)

            idx_state = meta_dict.setdefault('idx_state', {})
            idx_state.setdefault(str(state), []).append(trial_idx)

            # update the total trial count
            meta_dict['trial_n'] = max(int(meta_dict.get('trial_n', 0)), trial_idx + 1)
            meta_dict['subject_dict'][int(subject_id)] = subject_entry
            txn.put(b'meta', pickle.dumps(meta_dict), db = self.meta_db)

        return trial_idx

    # [Accessor] filter trials by subject_id, phase, state
    def filter_trials(self, subject_id: int = None, phase: str = None, state: str = None) -> list:
        """
        Performs ultra-fast O(1) set-intersection queries across the pre-compiled inverted indexes
        to return matching global trial indices.
        """
        meta_dict = self._read_meta_dict()
        idx_subjects = meta_dict.get('idx_subjects', {})
        idx_phase = meta_dict.get('idx_phase', {})
        idx_state = meta_dict.get('idx_state', {})

        # start from the first requested constraint and keep intersecting from there
        candidates = None

        if subject_id is not None:
            candidates = set(idx_subjects.get(int(subject_id), []))
        if phase is not None:
            phase_candidates = set(idx_phase.get(str(phase), []))
            candidates = phase_candidates if candidates is None else candidates.intersection(phase_candidates)
        if state is not None:
            state_candidates = set(idx_state.get(str(state), []))
            candidates = state_candidates if candidates is None else candidates.intersection(state_candidates)

        if candidates is None:
            candidates = set(range(int(meta_dict.get('trial_n', 0))))

        return sorted(list(candidates))

    # [Accessor] read a SINGLE modality
    def read_trial_modality(self, trial_idx: int, modality_name: str):
        """
        Reads and decodes a specific modality (e.g., 'eeg') for a trial.

        If resource 'storage' == 'lmdb':
        1. Fetches trial metadata to locate the database resource pointer.
        2. Retrieves the raw binary block using the mapped 'key'.
        3. Calculates total byte length using 'shape' and 'dtype'.
        4. Slices the binary data starting from 'offset'.
        5. Decodes and shapes it back into a NumPy array using 'dtype' and 'shape'.

        If resource 'storage' == 'disk':
        1. Accesses the file located at 'path'.
        2. Safely decodes or streams the segment starting from 'offset_index'.
        """
        trial_key = f"trial_{trial_idx}".encode('utf-8')
        with self.env.begin(write = False) as txn:
            # fetch the trial description first so we can locate the resource cleanly
            trial_blob = txn.get(trial_key, db = self.meta_db)
            if trial_blob is None:
                raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (read_trial_modality) Trial '{trial_key.decode('utf-8')}' does not exist.")
            trial_entry = pickle.loads(trial_blob)
            resource = trial_entry.get('data', {}).get(modality_name)
            if resource is None:
                raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (read_trial_modality) Modality '{modality_name}' for trial '{trial_idx}' not found.")

            # decode LMDB-backed resources directly from the raw buffer
            storage = resource.get('storage')
            if storage == 'lmdb':
                key = resource.get('key')
                if key is None:
                    raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (read_trial_modality) LMDB resource for modality '{modality_name}' is missing its data key.")
                raw_buffer = txn.get(key, db = self.data_db)
                if raw_buffer is None:
                    raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (read_trial_modality) Resource key '{key}' is missing from the data sub-database.")
                offset = int(resource.get('offset', 0))
                shape = tuple(resource.get('shape', ()))
                dtype = str(resource.get('dtype', 'uint8'))
                total_elements = int(np.prod(shape)) if len(shape) > 0 else 1
                total_bytes = int(total_elements * np.dtype(dtype).itemsize)
                payload = raw_buffer[offset : offset + total_bytes]
                return np.frombuffer(payload, dtype = np.dtype(dtype)).reshape(shape)

            # decode simple disk-backed NumPy resources where possible
            if storage == 'disk':
                disk_path = resource.get('path')
                if disk_path is None:
                    raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (read_trial_modality) Disk resource for modality '{modality_name}' is missing its path.")
                file_type = str(resource.get('type', '')).lower()
                offset_index = int(resource.get('offset_index', 0))

                if file_type in ['npy', 'npz']:
                    if file_type == 'npy':
                        array = np.load(disk_path)
                        return array[offset_index:] if offset_index > 0 else array
                    if file_type == 'npz':
                        array = np.load(disk_path)
                        if isinstance(array, np.lib.npyio.NpzFile):
                            keys = list(array.files)
                            if len(keys) == 0:
                                raise ValueError(f"\033[1;31m[Database] ERROR\033[0m: (read_trial_modality) '.npz' file '{disk_path}' does not contain any arrays.")
                            return array[keys[0]][offset_index:]
                        return array[offset_index:]

                raise NotImplementedError(f"\033[1;31m[Database] ERROR\033[0m: (read_trial_modality) Direct disk decoding for format '{file_type}' is not implemented yet. The resource path is '{disk_path}'.")

            raise ValueError(f"\033[1;31m[Database] ERROR\033[0m: (read_trial_modality) Unsupported resource storage type '{storage}' for modality '{modality_name}'.")

    # [Accessor] read all modalities for a trial
    def read_trial_modalities(self, trial_idx: int) -> dict:
        """
        Reads and decodes all modalities for a given trial.
        For the modality with a 'path' to actual data, we will try to read the data from file

        Returns:
            A dictionary mapping modality names to their corresponding NumPy arrays.
        """
        with self.env.begin(write = False) as txn:
            trial_blob = txn.get(f"trial_{trial_idx}".encode('utf-8'), db = self.meta_db)
            if trial_blob is None:
                raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (read_trial_modalities) Trial '{trial_idx}' not found.")
            trial_entry = pickle.loads(trial_blob)

        return {
            modality_name: self.read_trial_modality(trial_idx, modality_name)
            for modality_name in trial_entry.get('data', {}).keys()
        }

    # [Accessor] get a database component
    def get_db_attribute(self, path: str):
        """
        Retrieves a specific attribute from the database

        Arguments:
            path : str : dot-separated path to the attribute (e.g., 'meta.subject_dict')
                         <meta|data>.<key>.<subkey>... for nested access
        
        Returns:
            The value of the attribute at the specified path.
        """
        # parse the path
        components = path.split('.')
        # at least 2 components are needed: <meta|data>.<key>
        if len(components) < 2:
            raise ValueError(f"\033[1;31m[Database] ERROR\033[0m: (get_db_attribute) Invalid path '{path}'. Must be at least '<meta|data>.<key>'.")

        # get the database component
        db_name = components[0]
        if db_name not in ['meta', 'data']:
            raise ValueError(f"\033[1;31m[Database] ERROR\033[0m: (get_db_attribute) Invalid database '{db_name}'. Must be 'meta' or 'data'.")

        # Access the database
        db = self.meta_db if db_name == 'meta' else self.data_db

        # retrieve the value
        with self.env.begin(write = False) as txn:
            if db_name == 'meta':
                meta_dict = self._read_meta_dict()
                if components[1] in meta_dict:
                    value = meta_dict[components[1]]
                else:
                    key = components[1].encode('utf-8')
                    value = txn.get(key, db = db)
                    if value is None:
                        raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (get_db_attribute) Key '{components[1]}' not found in '{db_name}' database.")
                    value = pickle.loads(value)
            else:
                key = components[1].encode('utf-8')
                value = txn.get(key, db = db)
                if value is None:
                    raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (get_db_attribute) Key '{components[1]}' not found in '{db_name}' database.")

            # traverse nested keys if any
            for subkey in components[2:]:
                if isinstance(value, dict) and subkey in value:
                    value = value[subkey]
                else:
                    raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (get_db_attribute) Subkey '{subkey}' not found in the current context.")

        # return the final value
        return value

    # [Mutator] close the database
    def close(self):
        """
        Synchronizes and closes the LMDB environment securely.
        """
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
            self.env = None
        # verbose
        print(f"\033[1;34m[Database] INFO\033[0m: (close) Closed Database at '{self.db_path}'")

    # TO WORK WITH python 'with' block
    def __enter__(self):
        return self

    # TO WORK WITH python 'with' block
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # [ACCESSOR] Print a briefing of current database
    def print_brief(self):
        # get meta dict
        meta_dict = self._read_meta_dict()
        # print summary
        print(f"\033[1;34m[Database] INFO\033[0m: (print_brief) Database at '{self.db_path}'")
        # name
        print(f"  Name: {meta_dict.get('name', '')}")
        # trial count
        print(f"  Trial Count (total): {meta_dict.get('trial_n', 0)}")
        # subject count
        print(f"  Subject Count: {len(meta_dict.get('subject_dict', {}))}")
        # phase/split count/ratio
        idx_phase = meta_dict.get('idx_phase', {})
        print(f"  Phase/Split Count: {len(idx_phase)}")
        for phase, trials in idx_phase.items():
            print(f"    Phase '{phase}': {len(trials)} trials ({len(trials)/meta_dict.get('trial_n', 1):.2%})")
        # state/condition count/ratio
        idx_state = meta_dict.get('idx_state', {})
        print(f"  State/Condition Count: {len(idx_state)}")
        for state, trials in idx_state.items():
            print(f"    State '{state}': {len(trials)} trials ({len(trials)/meta_dict.get('trial_n', 1):.2%})")
        # done
        return

    # [INTERNAL] Load and store the root metadata dictionary
    def _read_meta_dict(self) -> dict:
        with self.env.begin(write = False) as txn:
            value = txn.get(b'meta', db = self.meta_db)
            if value is None:
                raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (_read_meta_dict) Root 'meta' key is missing.")
            return pickle.loads(value)

    # [INTERNAL] Write the root metadata dictionary
    def _write_meta_dict(self, meta_dict: dict):
        with self.env.begin(write = True) as txn:
            txn.put(b'meta', pickle.dumps(meta_dict), db = self.meta_db)

    # [INTERNAL] Hash resource keys
    def _hash_resource_key(self, data: bytes) -> bytes:
        return hashlib.md5(data).digest()

    # [INTERNAL] Store binary data in the 'data' sub-database and return the generated key
    def _store_data_blob(self, data: bytes) -> bytes:
        # use a content hash so identical resources shared across trials are stored once only
        hash_key = self._hash_resource_key(data)
        data_key = f"resource_{hash_key.hex()}".encode('utf-8')
        with self.env.begin(write = True) as txn:
            if txn.get(data_key, db = self.data_db) is None:
                txn.put(data_key, data, db = self.data_db)
        return data_key

    # [INTERNAL] Build a clean resource dict for a modality without forcing the user to manage database keys
    def _make_resource(self, resource: Any, *, modality_name: str, path: str = None, file_type: str = None, shape: tuple = None, dtype: str = None, offset: int = 0, storage: str = None) -> dict:
        """
        Builds a single modality/resource dict from a raw array, a file path, or an already-formatted dict.
        This is intentionally user-friendly: the caller does not have to manage LMDB keys or track the data DB mapping manually.
        """

        # if we received a dict
        if isinstance(resource, dict):
            resource_dict = dict(resource)
            # get or infer storage type
            if storage is not None:
                resource_dict['storage'] = storage
            if 'storage' not in resource_dict:
                if 'path' in resource_dict:
                    resource_dict['storage'] = 'disk'
                elif 'key' in resource_dict or 'value' in resource_dict or 'data' in resource_dict:
                    resource_dict['storage'] = 'lmdb'
                else:
                    raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (make_resource) Resource dict for modality '{modality_name}' is missing 'storage' and cannot be inferred.")
            if resource_dict.get('storage') == 'disk':
                if 'path' not in resource_dict and path is None:
                    raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (make_resource) Disk resource for modality '{modality_name}' does not contain a 'path'.")
                resource_dict['path'] = str(path if path is not None else resource_dict['path'])
                resource_dict['type'] = str(file_type or resource_dict.get('type') or os.path.splitext(resource_dict['path'])[1].lstrip('.'))
                resource_dict['offset_index'] = int(resource_dict.get('offset_index', offset))
                resource_dict.setdefault('shape', shape)
                resource_dict.setdefault('dtype', dtype)
                resource_dict['storage'] = 'disk'
                return resource_dict
            elif resource_dict.get('storage') == 'lmdb':
                resource_dict['offset'] = int(resource_dict.get('offset', offset))
                resource_dict.setdefault('shape', shape)
                resource_dict.setdefault('dtype', dtype)
                # pick the in-memory payload if the caller provided one
                value = resource_dict.get('value', resource_dict.get('data'))
                if 'key' not in resource_dict and value is not None:
                    resource_dict.pop('value', None)
                    resource_dict.pop('data', None)
                    data = self._serialise_array(value, shape = shape, dtype = dtype)
                    resource_dict['key'] = self._store_data_blob(data)
                return resource_dict
            else:
                raise ValueError(f"\033[1;31m[Database] ERROR\033[0m: (make_resource) Unsupported storage type '{resource_dict.get('storage')}' for modality '{modality_name}'.")

        # if we received a string or path-like object, treat it as a disk path
        if isinstance(resource, (str, os.PathLike)):
            # convert to absolute path
            file_path = os.fspath(resource)
            if path is not None:
                file_path = path
            # treat it as a disk resource when it looks like a real path
            is_disk_path = os.path.exists(file_path) or bool(os.path.splitext(file_path)[1])
            if is_disk_path:
                return {
                    'storage': 'disk',
                    'path': os.path.abspath(file_path),
                    'type': str(file_type or os.path.splitext(file_path)[1].lstrip('.')),
                    'offset_index': int(offset),
                    'shape': tuple(shape) if shape is not None else None,
                    'dtype': str(dtype) if dtype is not None else None
                }

        # treat anything else as raw data to be stored in LMDB
        data = self._serialise_array(resource, shape = shape, dtype = dtype)
        key = self._store_data_blob(data)

        # return the resource dict
        return {
            'storage': 'lmdb',
            'key': key,
            'offset': int(offset),
            'shape': tuple(shape) if shape is not None else self._infer_shape(resource),
            'dtype': str(dtype) if dtype is not None else self._infer_dtype(resource)
        }

    # [INTERNAL] Serialise an array-like object to raw bytes
    def _serialise_array(self, resource: Any, *, shape: tuple = None, dtype: str = None) -> bytes:
        if isinstance(resource, (bytes, bytearray, memoryview)):
            return bytes(resource)

        array = np.asarray(resource)
        if shape is not None:
            array = array.reshape(shape)
        if dtype is not None:
            array = array.astype(np.dtype(dtype), copy = False)
        return array.tobytes()

    # [INTERNAL] Infer array shape without forcing the user to provide it
    def _infer_shape(self, resource: Any):
        try:
            array = np.asarray(resource)
            return tuple(array.shape)
        except Exception:
            return None

    # [INTERNAL] Infer array dtype without forcing the user to provide it
    def _infer_dtype(self, resource: Any):
        try:
            array = np.asarray(resource)
            return str(array.dtype)
        except Exception:
            return None

    # [INTERNAL] Get new trial index, just peek, no increment
    def _get_new_trial_index(self) -> int:
        # read the current total trial count without opening a write transaction
        with self.env.begin(write = False) as txn:
            meta_dict = pickle.loads(txn.get(b'meta', db = self.meta_db))
            trial_n = meta_dict['trial_n']
        return int(trial_n)


# Entry point (testing)
# run with `python -m dataset.database`
if __name__ == "__main__":

    # import libraries for testing
    import shutil
    import tempfile

    # Create temp dir
    tmp = tempfile.mkdtemp()
    db_dir = os.path.join(tmp, "db")
    print("\033[1;34m[Database] INFO\033[0m: Starting LMDB Database Smoke Test")
    print(f"\033[1;34m[Database] INFO\033[0m: Temp Database Path: {db_dir}\n")

    try:
        # Instantiate the database (creates named DBs internally)
        db = Database(db_dir, mode = 'c', map_size = int(10 * 1024 * 1024))  # 10MB Map for test
        # set name
        db.write_meta_name("TestDB")
        print("\033[1;34m[Database] INFO\033[0m: Database initialized with dual sub-dbs (meta/data).")

        # Insert a trial with dummy data
        # prepare resources
        dummy_eeg = np.random.rand(800, 64).astype(np.float32)
        dummy_fmri_path = os.path.join(tmp, "dummy_fmri.npy")
        np.save(dummy_fmri_path, np.random.rand(64, 64, 64).astype(np.float32))  # Save dummy fmri data
        # assert that file saved
        assert os.path.exists(dummy_fmri_path), f"\033[1;31m[Database] ERROR\033[0m: Dummy fMRI file '{dummy_fmri_path}' was not created."
        resources = {
            'eeg': dummy_eeg,
            'fmri': {
                'path': dummy_fmri_path,
                'type': 'npy',
                'shape': (64, 64, 64),
                'dtype': 'float32'
            }
        }
        # insert trial
        trial_idx = db.insert_trial(subject_id = 1, phase = 'train', state = 'resting', resources = resources)
        print(f"\033[1;34m[Database] INFO\033[0m: Inserted trial {trial_idx} with dummy EEG and fMRI data.")

        # close the database
        db.close()

        # reopen as read-only and read back the trial
        db = Database(db_dir, mode = 'r', map_size = int(10 * 1024 * 1024))
        print("\033[1;34m[Database] INFO\033[0m: Reopened database in read-only mode.")
        # use subject filter to get trial indices
        trial_indices = db.filter_trials(subject_id = 1)
        # assert we have the trial we just inserted
        assert trial_idx in trial_indices, f"\033[1;31m[Database] ERROR\033[0m: Inserted trial {trial_idx} not found in filter results."
        # read back the EEG modality
        eeg_data = db.read_trial_modality(trial_idx, 'eeg')
        assert np.allclose(eeg_data, dummy_eeg), f"\033[1;31m[Database] ERROR\033[0m: Retrieved EEG data does not match the original."
        # read back the fMRI modality
        fmri_data = db.read_trial_modality(trial_idx, 'fmri')
        original_fmri_data = np.load(dummy_fmri_path)
        assert np.allclose(fmri_data, original_fmri_data), f"\033[1;31m[Database] ERROR\033[0m: Retrieved fMRI data does not match the original."
        # try read all modalities
        all_modalities = db.read_trial_modalities(trial_idx)
        assert 'eeg' in all_modalities and 'fmri' in all_modalities, f"\033[1;31m[Database] ERROR\033[0m: Retrieved modalities do not include both EEG and fMRI."
        # print a brief summary
        db.print_brief()

        # close the database
        db.close()

    finally:
        # Cleanup test directories safely
        shutil.rmtree(tmp, ignore_errors = True)
    
    print("\033[1;32m[Database] OK\033[0m: Test completed successfully.")
