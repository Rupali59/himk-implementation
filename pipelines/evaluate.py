"""
Load trained model and test data; compute HIMK train-test kernel; predict; report accuracy and confusion matrix.
"""

import sys
from pathlib import Path

import numpy as np
import yaml
import joblib
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.hog import extract_hog_from_video
from src.kernel.himk import himk_train_test, base_kernel_rbf
from src.classify.svm_himk import load_model
from src.data import get_data_loader


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "config.yaml", type=Path)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--cache-dir", type=Path, help="Use cached HoG for test (optional)")
    ap.add_argument("--export", type=Path, help="Export results to JSON file")
    args = ap.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    loader = get_data_loader(data_cfg)
    model_dir = args.model_dir or Path(data_cfg["model_dir"])
    cache_dir = Path(data_cfg.get("cache_dir", str(model_dir / "cache")))
    base_kernel_gamma = config["himk"].get("base_kernel_gamma", 0.01)

    # Load model
    clf, le = load_model(model_dir / "svm_himk.joblib")
    model = joblib.load(model_dir / "cdhmm.joblib")
    meta = joblib.load(model_dir / "meta.joblib")
    class_names = meta["class_names"]

    # Load train (for kernel: test vs train)
    train_paths, train_labels = loader.get_split("train")
    if not train_paths:
        print("No train data; cannot compute test-train kernel.")
        sys.exit(1)
    cache_train = cache_dir / "hog_train.joblib"
    if cache_train.exists():
        td = joblib.load(cache_train)
        X_train, _ = td["sequences"], td["labels"]
    else:
        X_train = [extract_hog_from_video(str(p)) for p in tqdm(train_paths, desc="HoG train")]
    y_train = np.array(train_labels)

    # Load test
    test_paths, test_labels = loader.get_split(args.split)
    if not test_paths:
        print("No test data for split:", args.split)
        sys.exit(1)
    cache_test = cache_dir / f"hog_{args.split}.joblib"
    if args.cache_dir and (Path(args.cache_dir) / f"hog_{args.split}.joblib").exists():
        td = joblib.load(Path(args.cache_dir) / f"hog_{args.split}.joblib")
        X_test, y_test = td["sequences"], np.array(td["labels"])
    elif cache_test.exists():
        td = joblib.load(cache_test)
        X_test, y_test = td["sequences"], np.array(td["labels"])
    else:
        X_test = [extract_hog_from_video(str(p)) for p in tqdm(test_paths, desc=f"HoG {args.split}")]
        y_test = np.array(test_labels)
        cache_test.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"sequences": X_test, "labels": list(y_test)}, cache_test)

    # HIMK test-train kernel (use same scale as training so SVM input is consistent)
    base_kernel = lambda x, y: base_kernel_rbf(x, y, gamma=base_kernel_gamma)
    K_test_train = himk_train_test(model, X_train, X_test, base_kernel)
    kernel_scale = meta.get("kernel_scale", 1.0)
    if kernel_scale and kernel_scale > 0:
        K_test_train = np.nan_to_num(K_test_train, nan=0.0, posinf=1.0, neginf=0.0) / kernel_scale

    # Predict
    pred_idx = clf.predict(K_test_train)
    pred_labels = le.inverse_transform(pred_idx)
    class_list = list(le.classes_)
    y_true_idx = np.array([class_list.index(y) for y in y_test])

    # Accuracy
    acc = (pred_idx == y_true_idx).mean() * 100
    print(f"Accuracy ({args.split}): {acc:.2f}%")

    # Confusion matrix (percentage)
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true_idx, pred_idx):
        cm[t, p] += 1
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100
    print("\nConfusion matrix (%):")
    print("Rows=true, Cols=pred")
    header = " ".join(f"{c:>12}" for c in class_names)
    print(f"             {header}")
    for i, name in enumerate(class_names):
        row = " ".join(f"{cm_pct[i, j]:>11.2f}%" for j in range(n_classes))
        print(f"{name:>12} {row}")
    print(f"\nOverall accuracy: {acc:.2f}%")

    if args.export:
        import json
        args.export.parent.mkdir(parents=True, exist_ok=True)
        export_data = {
            "split": args.split,
            "accuracy_pct": round(float(acc), 2),
            "class_names": class_names,
            "confusion_matrix_pct": cm_pct.tolist(),
            "confusion_matrix_counts": cm.tolist(),
        }
        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2)
        print("Exported results to", args.export)


if __name__ == "__main__":
    main()
