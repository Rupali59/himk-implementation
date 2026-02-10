"""
SVM with precomputed HIMK kernel for human activity recognition.
Uses sklearn.svm.SVC(kernel='precomputed') with Gram matrix from HIMK.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


def fit_svm_himk(
    K_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
) -> tuple[SVC, LabelEncoder]:
    """
    Fit SVM on precomputed kernel matrix K_train (n_train, n_train).
    y_train: string or int labels.
    Returns (fitted SVC, LabelEncoder for classes).
    """
    le = LabelEncoder()
    y = le.fit_transform(y_train)
    clf = SVC(kernel="precomputed", C=C, class_weight=class_weight)
    clf.fit(K_train, y)
    return clf, le


def predict_svm_himk(
    clf: SVC,
    le: LabelEncoder,
    K_test_train: np.ndarray,
) -> np.ndarray:
    """
    Predict using SVM; K_test_train is (n_test, n_train) kernel block.
    Returns integer indices; decode with le.inverse_transform for class labels.
    """
    pred = clf.predict(K_test_train)
    return pred


def save_model(clf: SVC, le: LabelEncoder, path: Union[str, Path]) -> None:
    """Save SVM and label encoder to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "le": le}, path)


def load_model(path: Union[str, Path]) -> tuple[SVC, LabelEncoder]:
    """Load SVM and label encoder from disk."""
    data = joblib.load(Path(path))
    return data["clf"], data["le"]
