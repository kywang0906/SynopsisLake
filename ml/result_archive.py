"""Package completed results without random writes on Databricks Volumes.

ZIP needs seeking when it is built. Build and check it on the driver's local
disk, then use only sequential writes and reads at the published destination.
This helper does not alter any result CSV or benchmark measurements.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile


_BUFFER_SIZE = 1024 * 1024


def _inventory(source: Path, destination: Path) -> list[tuple[Path, tuple[int, int]]]:
    files = []
    for path in sorted(source.rglob("*")):
        # Reject directory links too: they can escape the result tree or cycle.
        if path.is_symlink():
            raise ValueError(f"Result archives do not follow symbolic links: {path}")
        if path.is_file() and path != destination and path.suffix.lower() != ".zip":
            stat = path.stat()
            files.append((path, (stat.st_size, stat.st_mtime_ns)))
    return files


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def package_directory(
    source_dir: str | Path,
    output_name: str = "09b_comparison_results.zip",
    *,
    overwrite: bool = False,
) -> Path:
    """Return a verified ZIP next to the result files, using local ZIP staging.

    All existing ZIP files are excluded, including partial previous exports.
    ``output_name`` must be a single filename. Existing destinations are refused
    unless ``overwrite=True`` is explicitly requested (for export recovery).
    A failed publication raises and may leave an incomplete destination ZIP;
    scientific result files are never changed or deleted. Do not run this while
    other code is still writing the results.
    """
    if (not isinstance(output_name, str) or output_name in ("", ".", "..")
            or "/" in output_name or "\\" in output_name
            or "\x00" in output_name):
        raise ValueError("output_name must be a single filename, without path components")
    original = Path(source_dir).expanduser()
    if original.is_symlink():
        raise ValueError(f"Result archives do not follow symbolic links: {original}")
    source = original.resolve(strict=True)
    if not source.is_dir():
        raise NotADirectoryError(source)
    destination = source / output_name
    if destination.is_symlink():
        raise ValueError(f"Result archives do not overwrite symbolic links: {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Archive already exists: {destination}; use overwrite=True to replace it")
    files = _inventory(source, destination)

    # /tmp is driver-local on Databricks; TMPDIR may instead point at a Volume.
    with tempfile.TemporaryDirectory(prefix="synopsislake-archive-", dir="/tmp") as temp:
        staged = Path(temp) / "results.zip"
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, _ in files:
                archive.write(path, arcname=path.relative_to(source).as_posix())
        # Reopen only the local file, after the writer has closed its directory.
        with zipfile.ZipFile(staged, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise zipfile.BadZipFile(f"Local result archive failed CRC verification: {bad_member}")
        if files != _inventory(source, destination):
            raise RuntimeError("Result files changed while packaging; wait for all result writers to finish")
        expected = _digest(staged)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Archive already exists: {destination}; use overwrite=True to replace it")
        # No ZipFile, seek, append, rename, or metadata-copy operation on Volume.
        with staged.open("rb") as reader, destination.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=_BUFFER_SIZE)
        if _digest(destination) != expected:
            raise OSError(f"Result archive copy failed SHA256/size verification: {destination}")
    return destination
