# Script to download dataset from hugging face
# requires huggingface_hub library, install with: `pip install huggingface_hub`

from __future__ import annotations
import os
import sys
import time
from typing import List, Optional, Sequence

# try importing, as this dependency may not be installed with environment setup
try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError as e:
    raise ImportError(
        "The 'huggingface_hub' library is required to use this script. "
        "Please install it using 'pip install huggingface_hub'."
    ) from e

from requests.exceptions import RequestException


class HF_REPO:

    def __init__(
        self,
        repo_id: str,
        revision: str = "main",
        repo_type: str = "dataset",
        hf_home: Optional[str] = None,
        hf_endpoint: Optional[str] = None,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.repo_type = repo_type
        self.hf_endpoint = hf_endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        self.hf_home = hf_home or os.environ.get("HF_HOME", os.path.join(os.getcwd(), "cache", "hf_cache"))
        self.api = HfApi(endpoint = self.hf_endpoint)

        # Recommended environment settings
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"  # Increased timeout for large files
        os.environ["HF_ENDPOINT"] = self.hf_endpoint
        os.environ["HF_HOME"] = self.hf_home

    def get_file_tree(self) -> List[str]:
        try:
            files = self.api.list_repo_files(
                repo_id = self.repo_id,
                repo_type = self.repo_type,
                revision = self.revision,
            )
            return sorted(files)
        except Exception as e:
            print(f"\033[1;33m[HF_REPO] WARN\033[0m: Unable to list files for {self.repo_id}: {e}")
            return []

    def download(
        self,
        target_path: str,
        ignore_patterns: Optional[Sequence[str]] = None,
        keep_patterns: Optional[Sequence[str]] = None,
        max_retries: int = 10,
        base_delay: int = 5,
    ) -> None:
        
        # create target directory if it doesn't exist
        os.makedirs(target_path, exist_ok = True)

        # Determine effective ignore and keep patterns
        effective_ignore_patterns = list(ignore_patterns) if ignore_patterns is not None else None
        effective_keep_patterns = list(keep_patterns) if keep_patterns is not None else None

        print(f"\033[1;34m[HF_REPO] INFO\033[0m: Starting download for {self.repo_id} ({self.repo_type}) to {target_path}")

        for attempt in range(max_retries):
            try:
                snapshot_download(
                    repo_id = self.repo_id,
                    revision = self.revision,
                    repo_type = self.repo_type,
                    local_dir = target_path,
                    ignore_patterns = effective_ignore_patterns,
                    allow_patterns = effective_keep_patterns,
                    max_workers = 8,
                    endpoint = self.hf_endpoint,
                    cache_dir = self.hf_home,
                )
                # force flush stdout to ensure all messages are printed before exit
                sys.stdout.flush()
                print("\n\n\033[1;32m[HF_REPO] OK\033[0m: Download completed successfully!")
                return
            except (RequestException, Exception) as e:
                delay = base_delay * (2 ** attempt)
                print(f"\033[1;33m[HF_REPO] WARN\033[0m: Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"\033[1;33m[HF_REPO] WARN\033[0m: Retrying in {delay} seconds.")
                    time.sleep(delay)
                else:
                    print("\033[1;31m[HF_REPO] ERROR\033[0m: Max retries reached. Please check your network.")
                    raise RuntimeError(f"\033[1;31m[HF_REPO] ERROR\033[0m: Download failed for {self.repo_id}") from e


# Example usage
if __name__ == "__main__":

    # ref only
    raise NotImplementedError("\033[1;31m[HF_REPO] ERROR\033[0m: This script is not meant to be run directly.")

    repo = HF_REPO(
        repo_id = "your-org/your-dataset",
        repo_type = "dataset",
    )

    files = repo.get_file_tree()
    print(files[:10])

    repo.download(
        keep_patterns = ["train/*.parquet", "metadata.json"],
    )
