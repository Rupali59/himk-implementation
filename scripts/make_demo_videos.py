"""Create minimal demo AVI files so the pipeline can run without the full KTH dataset."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
# 160x120, ~4 sec at 25fps = 100 frames; for demo use 20 frames each
W, H = 160, 120
FPS = 25
N_FRAMES = 20


def write_avi(path: Path, n_frames: int = N_FRAMES, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
    for _ in range(n_frames):
        frame = (rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
        out.write(frame)
    out.release()


def main():
    data_dir = ROOT / "data" / "kth"
    for split, n_videos_per_class in [("train", 4), ("test", 2)]:
        for i, cls in enumerate(CLASSES):
            d = data_dir / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n_videos_per_class):
                write_avi(d / f"demo_{cls}_{j}.avi", seed=hash((split, cls, j)) % 2**31)
    print("Created demo videos under", data_dir)


if __name__ == "__main__":
    main()
