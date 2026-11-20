"""Regression checks for exporting results to a non-seekable Volume."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from ml import result_archive


class _SequentialStream:
    """Expose sequential I/O, but fail like Volume storage on seek/tell."""

    def __init__(self, stream, *, corrupt=False):
        self.stream = stream
        self.corrupt = corrupt

    def write(self, data):
        if self.corrupt and data:
            data = bytes([data[0] ^ 1]) + data[1:]
        return self.stream.write(data)

    def read(self, size=-1):
        return self.stream.read(size)

    def seek(self, *args):
        raise OSError(5, "Volume does not support seeking")

    def tell(self):
        raise OSError(5, "Volume does not support tell")

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)


class ResultArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "results"
        (self.source / "nested").mkdir(parents=True)
        (self.source / "summary.csv").write_text("method,count\nexact,12\n")
        (self.source / "nested" / "per_query.csv").write_text("qId,answer\n1,7\n")

    @contextmanager
    def _volume_io(self, destination, *, corrupt=False):
        original_open = Path.open
        modes = []

        def volume_open(path, mode="r", *args, **kwargs):
            stream = original_open(path, mode, *args, **kwargs)
            if path == destination:
                modes.append(mode)
                self.assertIn(mode, ("wb", "rb"))
                return _SequentialStream(stream, corrupt=corrupt and mode == "wb")
            return stream

        with patch.object(Path, "open", volume_open):
            yield modes

    def test_zip_is_built_and_crc_checked_locally_then_copied_sequentially(self):
        destination = self.source / "09b_comparison_results.zip"
        # No ZIP should ever be read or written through Volume's random I/O API.
        original_zip = zipfile.ZipFile
        zip_paths = []

        def local_zip(path, *args, **kwargs):
            zip_paths.append(Path(path))
            self.assertNotEqual(Path(path), destination)
            self.assertFalse(Path(path).is_relative_to(self.source))
            return original_zip(path, *args, **kwargs)

        with self._volume_io(destination) as modes, patch.object(zipfile, "ZipFile", local_zip):
            returned = result_archive.package_directory(self.source)
        self.assertEqual(returned, destination)
        self.assertEqual(modes, ["wb", "rb"])
        self.assertEqual(len(zip_paths), 2)  # Local writer closes before CRC reader.
        with original_zip(destination) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), {"summary.csv", "nested/per_query.csv"})
            self.assertEqual(archive.read("nested/per_query.csv"), b"qId,answer\n1,7\n")

    def test_old_and_partial_archives_are_excluded_and_overwrite_is_explicit(self):
        destination = self.source / "09b_comparison_results.zip"
        destination.write_bytes(b"partial previous export")
        (self.source / "nested" / "old.ZIP").write_bytes(b"old archive")
        with self.assertRaises(FileExistsError):
            result_archive.package_directory(self.source)
        self.assertEqual(destination.read_bytes(), b"partial previous export")
        result_archive.package_directory(self.source, overwrite=True)
        with zipfile.ZipFile(destination) as archive:
            self.assertEqual(set(archive.namelist()), {"summary.csv", "nested/per_query.csv"})

    def test_crc_failure_prevents_publication(self):
        destination = self.source / "09b_comparison_results.zip"
        with patch.object(zipfile.ZipFile, "testzip", return_value="summary.csv"):
            with self.assertRaisesRegex(zipfile.BadZipFile, "CRC verification"):
                result_archive.package_directory(self.source)
        self.assertFalse(destination.exists())

    def test_copy_failure_raises_and_retry_excludes_partial_zip(self):
        original = (self.source / "summary.csv").read_bytes()
        destination = self.source / "09b_comparison_results.zip"
        original_copy = result_archive.shutil.copyfileobj

        def partial_copy(reader, writer, length=0):
            if getattr(writer, "name", None) == str(destination):
                writer.write(reader.read(15))
                raise OSError(5, "Simulated Volume write failure")
            return original_copy(reader, writer, length)

        with patch.object(result_archive.shutil, "copyfileobj", partial_copy):
            with self.assertRaisesRegex(OSError, "Volume write failure"):
                result_archive.package_directory(self.source)
        self.assertTrue(destination.exists())
        self.assertEqual((self.source / "summary.csv").read_bytes(), original)
        result_archive.package_directory(self.source, overwrite=True)
        with zipfile.ZipFile(destination) as archive:
            self.assertNotIn(destination.name, archive.namelist())
            self.assertIsNone(archive.testzip())

    def test_corrupt_sequential_copy_is_rejected_by_readback(self):
        destination = self.source / "09b_comparison_results.zip"
        with self._volume_io(destination, corrupt=True):
            with self.assertRaisesRegex(OSError, "SHA256/size verification"):
                result_archive.package_directory(self.source)

    def test_output_names_cannot_escape_result_directory(self):
        for name in ("", ".", "..", "../elsewhere.zip", "/tmp/elsewhere.zip", "nested/file.zip", "..\\file.zip"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                result_archive.package_directory(self.source, name)

    def test_symlink_escape_cycle_and_destination_are_rejected(self):
        outside = Path(self.temp.name) / "outside.txt"
        outside.write_text("outside")
        for name, target in (("escape.txt", outside), ("cycle", self.source),
                             ("09b_comparison_results.zip", outside)):
            path = self.source / name
            path.symlink_to(target)
            try:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "symbolic links"):
                    result_archive.package_directory(self.source, overwrite=True)
            finally:
                path.unlink()
        link = Path(self.temp.name) / "result-link"
        link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            result_archive.package_directory(link)

    def test_changing_source_results_fail_before_publication(self):
        original_testzip = zipfile.ZipFile.testzip

        def mutate_after_crc(archive):
            result = original_testzip(archive)
            (self.source / "summary.csv").write_text("changed while exporting\n")
            return result

        with patch.object(zipfile.ZipFile, "testzip", mutate_after_crc):
            with self.assertRaisesRegex(RuntimeError, "changed while packaging"):
                result_archive.package_directory(self.source)
        self.assertFalse((self.source / "09b_comparison_results.zip").exists())


if __name__ == "__main__":
    unittest.main()
