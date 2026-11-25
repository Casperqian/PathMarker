"""Open whole-slide images through the backend required by their format."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from PIL import Image


def _open_openslide(path: Path):
    import openslide

    return openslide.OpenSlide(str(path))


def _open_aslide(path: Path):
    try:
        from Aslide import Slide
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "KFB support requires Aslide in the active Python environment. "
            "Install the local vendor package into the sam3 environment."
        ) from error
    return Slide(str(path))


class _RecoveringOpenSlide:
    """Reopen after a corrupt JPEG tile and blank only the unreadable region."""

    def __init__(self, path: Path):
        self._path = path
        self._slide = _open_openslide(path)
        self._reopen_lock = threading.Lock()
        self._warned_locations: set[tuple[int, int, int, tuple[int, int]]] = set()

    def __getattr__(self, name: str):
        return getattr(self._slide, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self) -> None:
        self._slide.close()

    @staticmethod
    def _is_corrupt_jpeg_error(error: Exception) -> bool:
        return "Corrupt JPEG data" in str(error)

    def _reopen_after_error(self, failed_slide):
        with self._reopen_lock:
            if self._slide is failed_slide:
                failed_slide.close()
                self._slide = _open_openslide(self._path)
            return self._slide

    def read_region(self, location, level: int, size: tuple[int, int]):
        current_slide = self._slide
        try:
            return current_slide.read_region(location, level, size)
        except Exception as error:
            if not self._is_corrupt_jpeg_error(error):
                raise

        reopened_slide = self._reopen_after_error(current_slide)
        try:
            return reopened_slide.read_region(location, level, size)
        except Exception as error:
            if not self._is_corrupt_jpeg_error(error):
                raise

        warning_key = (int(location[0]), int(location[1]), int(level), tuple(map(int, size)))
        if warning_key not in self._warned_locations:
            self._warned_locations.add(warning_key)
            print(
                f"Skipping unreadable JPEG region in {self._path}: "
                f"location={location}, level={level}, size={size}",
                file=sys.stderr,
            )
        return Image.new("RGBA", size, "white")


def open_slide(path: Path):
    """Return an OpenSlide-compatible reader for an SVS/TIFF/KFB slide."""
    path = Path(path)
    if path.suffix.lower() == ".kfb":
        return _open_aslide(path)
    return _RecoveringOpenSlide(path)
