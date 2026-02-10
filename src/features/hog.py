"""
Histogram of Oriented Gradients (HoG) feature extraction from video frames.

Per-frame HoG as in the thesis: each frame is subdivided into cells/blocks,
HoG is computed per cell and concatenated to form a D-dimensional vector per frame.
Video is represented as a sequence of these vectors (variable length).
Frames are resized to 160x120 to match thesis (KTH downsampled resolution).
"""

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np

try:
    from skimage.feature import hog as skimage_hog
except ImportError:
    skimage_hog = None

# Default HoG params: cell 8x8, block 2x2 cells, 9 orientation bins (Dalal-Triggs style)
DEFAULT_CELL_SIZE = (8, 8)
DEFAULT_BLOCK_SIZE = (2, 2)  # in cells
DEFAULT_NBINS = 9
TARGET_SIZE = (160, 120)  # thesis: 160x120 (height, width)


def extract_hog_frame(
    frame: np.ndarray,
    cell_size: tuple = DEFAULT_CELL_SIZE,
    block_size: tuple = DEFAULT_BLOCK_SIZE,
    nbins: int = DEFAULT_NBINS,
    target_size: tuple = TARGET_SIZE,
) -> np.ndarray:
    """
    Extract HoG descriptor for a single frame.
    Frame is resized to target_size (160x120) then HoG is computed.
    Returns a 1D vector of shape (D,).
    """
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = target_size[0], target_size[1]
    if frame.shape[0] != h or frame.shape[1] != w:
        frame = cv2.resize(frame, (w, h))
    frame = frame.astype(np.float64) / 255.0 if frame.dtype == np.uint8 else frame

    if skimage_hog is not None:
        # skimage: orientations=nbins, pixels_per_cell=cell_size, cells_per_block=block_size
        desc = skimage_hog(
            frame,
            orientations=nbins,
            pixels_per_cell=cell_size,
            cells_per_block=block_size,
            feature_vector=True,
            channel_axis=None,
        )
        return desc.astype(np.float64)
    # Fallback: simple gradient histogram (fewer bins to keep D manageable)
    gx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx) * 180 / np.pi
    ang = (ang + 180) / 360 * nbins
    ang = np.clip(ang.astype(int), 0, nbins - 1)
    hist = np.bincount(ang.flatten(), minlength=nbins, weights=mag.flatten())
    hist = hist / (hist.sum() + 1e-10)
    return hist.astype(np.float64)


def extract_hog_sequence(
    frames: List[np.ndarray],
    cell_size: tuple = DEFAULT_CELL_SIZE,
    block_size: tuple = DEFAULT_BLOCK_SIZE,
    nbins: int = DEFAULT_NBINS,
    target_size: tuple = TARGET_SIZE,
) -> np.ndarray:
    """
    Extract HoG for a sequence of frames. Returns array of shape (T, D).
    """
    if not frames:
        return np.zeros((0, 0), dtype=np.float64)
    descs = [
        extract_hog_frame(f, cell_size, block_size, nbins, target_size)
        for f in frames
    ]
    return np.array(descs, dtype=np.float64)


def video_to_frames(video_path: Union[str, Path]) -> List[np.ndarray]:
    """Read a video file and return list of frames (BGR)."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_hog_from_video(
    video_path: Union[str, Path],
    cell_size: tuple = DEFAULT_CELL_SIZE,
    block_size: tuple = DEFAULT_BLOCK_SIZE,
    nbins: int = DEFAULT_NBINS,
    target_size: tuple = TARGET_SIZE,
) -> np.ndarray:
    """
    Load video and return HoG sequence of shape (T, D).
    """
    frames = video_to_frames(video_path)
    return extract_hog_sequence(
        frames, cell_size, block_size, nbins, target_size
    )


def extract_hog_from_video_dir(
    video_dir: Union[str, Path],
    pattern: str = "*.avi",
    cell_size: tuple = DEFAULT_CELL_SIZE,
    block_size: tuple = DEFAULT_BLOCK_SIZE,
    nbins: int = DEFAULT_NBINS,
    target_size: tuple = TARGET_SIZE,
) -> dict:
    """
    Extract HoG from all videos in a directory.
    Returns dict: { video_stem: array of shape (T, D) }.
    """
    video_dir = Path(video_dir)
    out = {}
    for p in sorted(video_dir.glob(pattern)):
        try:
            out[p.stem] = extract_hog_from_video(
                p, cell_size, block_size, nbins, target_size
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process {p}: {e}") from e
    return out
