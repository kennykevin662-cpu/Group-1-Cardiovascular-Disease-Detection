"""THÀNH VIÊN 1 – ECG
Bước 4-6: xây dựng, huấn luyện, đánh giá, lưu mô hình
Output: ecg_model.pkl (và ecg_scaler.pkl nếu có)
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42

# TODO 4: Xây dựng và huấn luyện mô hình Machine Learning / Deep Learning
def build_model():
    # Random Forest: ổn định với đặc trưng ECG dạng bảng, không cần chuẩn hoá,
    # class_weight="balanced" để bù lệch lớp (bất thường thường ít hơn/nhiều hơn bình thường)
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
 
def train(model, X_train, y_train, X_val, y_val):
    # Random Forest không có "loss theo epoch" nên theo dõi accuracy train vs val
    # để phát hiện overfitting (train cao, val thấp = overfit)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"[TRAIN] Accuracy train: {train_acc:.4f} | Accuracy val: {val_acc:.4f}")
    return model


# TODO 5: Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Nhị phân (bình thường/bất thường) -> "binary"; nhiều lớp -> "weighted"
    avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=avg, zero_division=0),
    }
    for name, value in metrics.items():
        print(f"[TEST] {name:<9}: {value:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("[TEST] Confusion matrix:\n", cm)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Lưu hình confusion matrix để chèn vào báo cáo
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title("ECG – Confusion Matrix")
    out_png = os.path.join(BASE_DIR, "ecg_confusion_matrix.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[TEST] Đã lưu hình: {out_png}")

    return metrics, cm



# TODO 6: Lưu mô hình đã huấn luyện và viết báo cáo phần ECG
def save_model(model, path="mohinh/ECG/ecg_model.pkl"):
    # Lưu bằng joblib; predict.py chỉ cần joblib.load(path)
    # Giữ đường dẫn tương đối như skeleton; nếu chạy từ thư mục khác thì dùng BASE_DIR
    if not os.path.isabs(path):
        # nếu đang chạy ngay trong mohinh/ECG thì lưu cạnh file này
        path = os.path.join(BASE_DIR, os.path.basename(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[SAVE] Đã lưu mô hình: {path}")
    # Báo cáo ECG (viết ở file riêng): mục tiêu, dữ liệu PTB-XL, đặc trưng, mô hình RF,
    # kết quả 4 chỉ số + confusion matrix, hạn chế.


if __name__ == "__main__":
    # Đổi đường dẫn nếu file .npy nằm chỗ khác (ví dụ dataset/)
    X = np.load(os.path.join(BASE_DIR, "X_ecg.npy"))
    y = np.load(os.path.join(BASE_DIR, "y_ecg.npy"))
    print(f"X: {X.shape} | y: {y.shape} | phân bố nhãn: {np.bincount(y.astype(int))}")

    # 70% train / 15% val / 15% test, chia theo tỉ lệ nhãn (stratify)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
    )

    model = build_model()
    model = train(model, X_train, y_train, X_val, y_val)
    evaluate(model, X_test, y_test)
    save_model(model)

