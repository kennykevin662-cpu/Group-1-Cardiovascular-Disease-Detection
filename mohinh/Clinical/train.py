"""THÀNH VIÊN 3 – Clinical
Bước 4-6: xây dựng, huấn luyện, đánh giá, lưu mô hình
Output: clinical_model.pkl
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from preprocess import X_PATH, Y_PATH, build_dataset

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "clinical_model.pkl"
RANDOM_STATE = 42


# TODO 4: Xây dựng mô hình phân loại nguy cơ/bệnh tim theo nhãn dataset
def build_model():
    """Trả về dict {tên: (mô hình, lưới tham số)} để so sánh."""
    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 300], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        ),
    }
    try:  # XGBoost là tùy chọn: pip install xgboost
        from xgboost import XGBClassifier
        candidates["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
            {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        )
    except ImportError:
        print("(Bỏ qua XGBoost vì chưa cài: pip install xgboost)")
    return candidates


def train(model, X_train, y_train):
    """Cross-validation + tinh chỉnh tham số (GridSearch) cho từng mô hình.
    Trả về (mô hình tốt nhất đã fit, tên, bảng kết quả CV)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, best_name, best_score, best_est = {}, None, -1.0, None
    for name, (estimator, grid) in model.items():
        gs = GridSearchCV(estimator, grid, cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
        results[name] = {"cv_f1": gs.best_score_, "best_params": gs.best_params_}
        print(f"{name:20s} CV F1 = {gs.best_score_:.4f}  {gs.best_params_}")
        if gs.best_score_ > best_score:
            best_name, best_score, best_est = name, gs.best_score_, gs.best_estimator_
    print(f"\n=> Mô hình tốt nhất: {best_name}")
    return best_est, best_name, results


# TODO 5: Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print("\n--- Đánh giá trên tập test ---")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k:10s}: {metrics[k]:.4f}")
    print("Confusion matrix (hàng = thực tế, cột = dự đoán):")
    print(np.array(metrics["confusion_matrix"]))
    return metrics


# TODO 6: Lưu mô hình (báo cáo Clinical viết riêng bên Word)
def save_model(model, path=MODEL_PATH):
    path = Path(path)
    joblib.dump(model, path)
    print(f"Đã lưu mô hình: {path}")


if __name__ == "__main__":
    # load X_clinical.npy, y_clinical.npy -> build -> train -> evaluate -> save
    if not (X_PATH.exists() and Y_PATH.exists()):
        print("Chưa có X_clinical.npy / y_clinical.npy -> chạy preprocess trước...")
        build_dataset(explore=False)
    X, y = np.load(X_PATH), np.load(Y_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    models = build_model()
    best, best_name, cv_results = train(models, X_train, y_train)
    metrics = evaluate(best, X_test, y_test)
    save_model(best, MODEL_PATH)