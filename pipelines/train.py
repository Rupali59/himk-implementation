"""
Full training pipeline: load/extract HoG features -> train CDHMM -> HIMK Gram -> fit SVM -> save.
"""

import sys
from pathlib import Path

import numpy as np
import yaml
import joblib
from tqdm import tqdm

# Add implementation root so we can import src
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.hog import extract_hog_from_video, TARGET_SIZE
from src.kernel.himk import (
    build_himk_model_and_gram,
    himk_train_test,
    base_kernel_rbf,
)
from src.classify.svm_himk import fit_svm_himk, save_model
from src.data import get_data_loader


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_features_for_paths(paths: list, labels: list, cache_path=None):
    if cache_path and Path(cache_path).exists():
        data = joblib.load(cache_path)
        return data["sequences"], data["labels"]
    sequences = []
    for p in tqdm(paths, desc="HoG extraction"):
        seq = extract_hog_from_video(str(p) if not isinstance(p, (str, Path)) else p)
        if len(seq) == 0:
            seq = np.zeros((1, 1))  # placeholder to avoid empty
        sequences.append(seq)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"sequences": sequences, "labels": list(labels)}, cache_path)
    return sequences, list(labels)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "config.yaml", type=Path)
    ap.add_argument("--data-dir", type=Path, help="Override config data dir")
    ap.add_argument("--out-dir", type=Path, help="Override model output dir")
    args = ap.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    loader = get_data_loader(data_cfg)
    out_dir = args.out_dir or Path(data_cfg["model_dir"])
    cache_dir = Path(data_cfg.get("cache_dir", str(out_dir / "cache")))
    n_states = config["himk"].get("n_states", 15)
    n_mix = config["himk"].get("n_mix", 3)
    base_kernel_gamma = config["himk"].get("base_kernel_gamma", 0.01)
    svm_c = config["svm"].get("C", 1.0)
    random_state = config.get("random_state", 42)

    # Load train split (from filesystem or DB)
    train_paths, train_labels = loader.get_split("train")
    if not train_paths:
        print("No training data found.")
        print("  Filesystem: set data_dir and ensure data_dir/train/<class>/*.avi exist, or run scripts/download_kth.py")
        print("  MongoDB: set data.source to 'mongodb' and data.mongodb.* in config.")
        sys.exit(1)

    # Extract (or load cached) HoG
    cache_train = cache_dir / "hog_train.joblib"
    X_train, y_train = extract_features_for_paths(train_paths, train_labels, cache_train)
    y_train = np.array(y_train)

    # Subsample if huge (for speed)
    max_train = config.get("max_train_samples")
    if max_train and len(X_train) > max_train:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_train), size=max_train, replace=False)
        X_train = [X_train[i] for i in idx]
        y_train = y_train[idx]

    # Train CDHMM and compute HIMK Gram
    print("Training CDHMM and computing HIMK Gram...")
    base_kernel = lambda x, y: base_kernel_rbf(x, y, gamma=base_kernel_gamma)
    model, K_train = build_himk_model_and_gram(
        X_train, y_train,
        n_states=n_states,
        n_mix=n_mix,
        base_kernel=base_kernel,
        random_state=random_state,
    )
    # Ensure finite, well-scaled kernel matrix for SVM (avoids sklearn matmul overflow/divide-by-zero)
    K_train = np.nan_to_num(K_train, nan=0.0, posinf=1.0, neginf=0.0)
    kernel_scale = float(np.abs(K_train).max()) or 1.0
    if kernel_scale > 0:
        K_train = K_train / kernel_scale
    np.fill_diagonal(K_train, np.maximum(np.diag(K_train), 1e-6))

    # Fit SVM
    print("Fitting SVM...")
    clf, le = fit_svm_himk(K_train, y_train, C=svm_c)

    # Save (include kernel_scale so evaluate can normalize test-train kernel the same way)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(clf, le, out_dir / "svm_himk.joblib")
    joblib.dump(model, out_dir / "cdhmm.joblib")
    joblib.dump({"class_names": list(le.classes_), "kernel_scale": kernel_scale}, out_dir / "meta.joblib")
    print("Model saved to", out_dir)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
    main()
