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
import hashlib # for resource key generation


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

        # ...

    # [Mutator] update meta-data 'name'
    def write_meta_name(self, name: str):
        """
        In the 'meta' sub-database, sets 'meta'['name'] = name.
        Our database will count the 'subject_dict', 'trail_n' and
        the inverted index lists automatically when you write trial entries. 
        """
        pass

    # [Accessor] filter trials by subject_id, phase, state
    def filter_trials(self, subject_id: int = None, phase: str = None, state: str = None) -> list:
        """
        Performs ultra-fast O(1) set-intersection queries across the pre-compiled inverted indexes
        to return matching global trial indices.
        """
        pass

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
        pass

    # [Accessor] read all modalities for a trial
    def read_trial_modalities(self, trial_idx: int) -> dict:
        """
        Reads and decodes all modalities for a given trial.
        For the modality with a 'path' to actual data, we will try to read the data from file

        Returns:
            A dictionary mapping modality names to their corresponding NumPy arrays.
        """
        pass

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
            # get the top-level key
            key = components[1].encode('utf-8')
            value = txn.get(key, db = db)
            if value is None:
                raise KeyError(f"\033[1;31m[Database] ERROR\033[0m: (get_db_attribute) Key '{components[1]}' not found in '{db_name}' database.")

            # decode if it's the meta database
            if db_name == 'meta':
                value = pickle.loads(value)

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
        pass

    # TO WORK WITH python 'with' block
    def __enter__(self):
        return self

    # TO WORK WITH python 'with' block
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # [INTERNAL] Hash resource keys
    def _hash_resource_key(self, data: bytes) -> bytes:
        return hashlib.md5(data).digest()

    # [INTERNAL] Get new trial index, just peek, no increment
    def _get_new_trial_index(self) -> int:
        # load meta trail count
        with self.env.begin(write = True) as txn:
            meta_dict = pickle.loads(txn.get(b'meta', db = self.meta_db))
            trial_n = meta_dict['trial_n']
        return trial_n


# Entry point (testing)
# run with `python -m dataset.database`
if __name__ == "__main__":
    pass
