"""THÀNH VIÊN – PPG | Clinical Database Classification
Bước 4–6: xây dựng, huấn luyện, đánh giá và lưu mô hình Random Forest.
Đầu vào: X_ppg.npy, y_ppg.npy. Đầu ra: ppg_model.pkl, ma trận nhầm lẫn.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    from .preprocess import FEATURE_NAMES
except ImportError:
    from preprocess import FEATURE_NAMES

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42


def build_model():
    """Pipeline điền giá trị thiếu (nếu có) và Random Forest phân loại."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ))
    ])


def train(model, X_train, y_train, X_val=None, y_val=None):
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"[TRAIN] Accuracy: {train_acc:.4f}")
    if X_val is not None and y_val is not None:
        print(f"[VALIDATION] Accuracy: {accuracy_score(y_val, model.predict(X_val)):.4f}")
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    print("\n===== KẾT QUẢ ĐÁNH GIÁ CLINICAL =====")
    for name, value in metrics.items():
        print(f"{name:<10}: {value:.4f}")
    print("\nClassification report:\n", classification_report(
        y_test, y_pred, target_names=["Không bệnh", "Có bệnh"], zero_division=0
    ))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion matrix:\n", cm)
    ConfusionMatrixDisplay(cm, display_labels=["Không bệnh", "Có bệnh"]).plot(cmap="Blues", values_format="d")
    plt.title("PPG folder – Clinical Classification")
    plt.tight_layout()
    image_path = BASE_DIR / "ppg_confusion_matrix.png"
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu ma trận nhầm lẫn: {image_path}")
    return metrics, cm


def save_model(model, path=None):
    path = Path(path) if path else BASE_DIR / "ppg_model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "classes": {0: "Không mắc bệnh", 1: "Có bệnh"},
        "dataset": "UCI Heart Disease – processed Cleveland",
        "task": "Clinical database binary classification",
        "model_name": "RandomForestClassifier"
    }
    joblib.dump(bundle, path)
    print(f"[SAVE] Đã lưu mô hình: {path}")
    return path


def main():
    x_path, y_path = BASE_DIR / "X_ppg.npy", BASE_DIR / "y_ppg.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError("Chưa có X_ppg.npy/y_ppg.npy. Hãy chạy preprocess.py trước.")
    X, y = np.load(x_path), np.load(y_path)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or len(X) != len(y):
        raise ValueError("Kích thước dữ liệu không khớp với danh sách đặc trưng.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1875, stratify=y_train, random_state=RANDOM_STATE
    )  # xấp xỉ 65% train, 15% validation, 20% test
    model = train(build_model(), X_train, y_train, X_val, y_val)
    metrics, cm = evaluate(model, X_test, y_test)
    save_model(model)
    report = {
        "dataset": "UCI Heart Disease – processed Cleveland",
        "task": "Clinical database binary classification",
        "n_samples": int(len(y)), "n_features": int(X.shape[1]),
        "split": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test))},
        "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "metrics": metrics, "confusion_matrix": cm.tolist(), "features": FEATURE_NAMES
    }
    with open(BASE_DIR / "ppg_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu báo cáo: {BASE_DIR / 'ppg_metrics.json'}")


if __name__ == "__main__":
    main()
