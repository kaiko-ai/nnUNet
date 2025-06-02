"""Model hub related IO operations."""

import os
import sys
from typing import Any, Dict

import torch
from fsspec.core import url_to_fs
from torch import hub
from torchvision.datasets import utils
from tqdm import tqdm


def load_state_dict_from_url(
    url: str,
    *,
    model_dir: str | None = None,
    filename: str | None = None,
    progress: bool = True,
    md5: str | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Loads the Torch serialized object at the given URL.

    If the object is already present and valid in `model_dir`, it's
    deserialized and returned.

    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where
    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        url: URL of the object to download.
        model_dir: Directory in which to save the object.
        filename: Name for the downloaded file. Filename from ``url`` will be used if not set.
        progress: Whether or not to display a progress bar to stderr.
        md5: MD5 file code to check whether the file is valid. If not, it will re-download it.
        force: Whether to download the file regardless if it exists.
    """
    model_dir = model_dir or os.path.join(hub.get_dir(), "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, filename or os.path.basename(url))
    if force or not utils.check_integrity(cached_file, md5):
        sys.stderr.write(f"Downloading: '{url}' to {cached_file}\n")
        _download_url_to_file(url, cached_file, progress=progress)
        if md5 is None or not utils.check_integrity(cached_file, md5):
            sys.stderr.write(f"File MD5: {utils.calculate_md5(cached_file)}\n")

    return torch.load(cached_file, map_location="cpu")


def _download_url_to_file(
    url: str,
    dst: str,
    *,
    progress: bool = True,
) -> None:
    """Download object at the given URL to a local path.

    Args:
        url: URL of the object to download.
        dst: Full path where object will be saved.
        chunk_size: The size of each chunk to read in bytes.
        progress: Whether or not to display a progress bar to stderr.
    """
    try:
        _download_with_fsspec(url=url, dst=dst, progress=progress)
    except Exception:
        try:
            hub.download_url_to_file(url=url, dst=dst, progress=progress)
        except Exception as hub_e:
            raise RuntimeError(
                f"Failed to download file from {url} using both fsspec and hub."
            ) from hub_e


def _download_with_fsspec(
    url: str,
    dst: str,
    *,
    chunk_size: int = 1024 * 1024,
    progress: bool = True,
) -> None:
    """Download object at the given URL to a local path using fsspec.

    Args:
        url: URL of the object to download.
        dst: Full path where object will be saved.
        chunk_size: The size of each chunk to read in bytes.
        progress: Whether or not to display a progress bar to stderr.
    """
    filesystem, _ = url_to_fs(url, anon=False)
    total_size_bytes = filesystem.size(url)
    with (
        filesystem.open(url, "rb") as remote_file,
        tqdm(
            total=total_size_bytes,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            disable=not progress,
        ) as pbar,
    ):
        with open(dst, "wb") as local_file:
            while True:
                data = remote_file.read(chunk_size)
                if not data:
                    break

                local_file.write(data)
                pbar.update(chunk_size)
