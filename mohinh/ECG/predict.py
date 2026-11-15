"""THÀNH VIÊN 1 – ECG | Dự đoán bất thường tim mạch bằng mô hình ecg_model.pkl.

Đã được hoàn thiện & sửa lỗi bởi Thành viên 4 (thay thế file trùng lặp với preprocess.py).
Hỗ trợ:
- predict(signal, fs=100): dự đoán trực tiếp từ mảng numpy (số_mẫu, số_chuyển_đạo)
- predict_record(record_path): dự đoán từ file WFDB hoặc tệp .npy/.csv
- Chạy trực tiếp qua dòng lệnh (CLI demo).
"""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np

# Import các hàm tiền xử lý và trích đặc trưng từ preprocess.py
try:
    from preprocess import FS, extract_features, preprocess_signal
except ImportError:
    from mohinh.ECG.preprocess import FS, extract_features, preprocess_signal

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "ecg_model.pkl"

LABEL_MAPPING = {
    0: "Bình thường (Normal ECG)",
    1: "Bất thường (Cardiovascular Abnormality)",
}


def load_model(model_path=MODEL_PATH):
    """Tải mô hình RandomForestClassifier đã được huấn luyện."""
    path = Path(model_path)
    if not path.is_absolute():
        path = HERE / path
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình ECG tại: {path}")
    return joblib.load(path)


def predict(signal: np.ndarray, fs: int = FS, model=None) -> dict:
    """Dự đoán bất thường tim mạch từ tín hiệu ECG.
    
    Args:
        signal: Mảng numpy chứa tín hiệu ECG. Kích thước lý tưởng là (số_mẫu, 12).
                Nếu là tín hiệu 1 kênh (N,) hoặc (N, 1), sẽ tự động nhân bản thành 12 kênh
                để tương thích kích thước đầu vào của mô hình.
        fs: Tần số lấy mẫu (mặc định 100 Hz).
        model: Đối tượng mô hình đã tải trước đó (tùy chọn).
        
    Returns:
        dict: Kết quả chẩn đoán gồm prediction, label, probability, confidence, heart_rate, r_peaks.
    """
    if model is None:
        model = load_model()

    sig = np.asarray(signal, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)

    # Đảm bảo có đủ 12 chuyển đạo tương thích với mô hình đã huấn luyện
    if sig.shape[1] < 12:
        reps = int(np.ceil(12 / sig.shape[1]))
        sig = np.tile(sig, (1, reps))[:, :12]
    elif sig.shape[1] > 12:
        sig = sig[:, :12]

    # Trích xuất 89 đặc trưng
    features = extract_features(sig, fs=fs)
    features_2d = features.reshape(1, -1)

    # Dự đoán
    pred = int(model.predict(features_2d)[0])
    proba = model.predict_proba(features_2d)[0]
    prob_abnormal = float(proba[1]) if len(proba) > 1 else float(pred)
    confidence = float(proba[pred]) if len(proba) > pred else 1.0

    # Trích xuất nhịp tim và số đỉnh R từ 5 đặc trưng cuối cùng (lead II)
    # 5 đặc trưng cuối: [HR, mean RR, std RR, RMSSD, len(peaks)]
    estimated_hr = float(features[-5])
    rmssd = float(features[-2])
    r_peaks_count = int(features[-1])

    return {
        "prediction": pred,
        "label": LABEL_MAPPING.get(pred, "Không xác định"),
        "probability": round(prob_abnormal, 4),
        "confidence": round(confidence, 4),
        "estimated_hr": round(estimated_hr, 1) if estimated_hr > 0 else None,
        "rmssd": round(rmssd, 4) if rmssd > 0 else None,
        "r_peaks_count": r_peaks_count,
        "features": features,
    }


def predict_record(record_path: str, fs: int = FS, model=None) -> dict:
    """Dự đoán từ đường dẫn tệp tín hiệu (.npy, .csv hoặc WFDB record)."""
    p = Path(record_path)
    if p.suffix == ".npy":
        sig = np.load(p)
    elif p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        sig = df.to_numpy()
    else:
        # Thử đọc qua thư viện wfdb nếu có
        import wfdb
        record_base = str(p.with_suffix(""))
        sig, fields = wfdb.rdsamp(record_base)
        fs = fields.get("fs", fs)

    res = predict(sig, fs=fs, model=model)
    res["record_path"] = str(p)
    return res


def main():
    """Demo dự đoán từ dòng lệnh."""
    ap = argparse.ArgumentParser(description="Chạy dự đoán ECG với mô hình ecg_model.pkl")
    ap.add_argument("--file", help="Đường dẫn tới tệp ECG (.npy, .csv, hoặc bản ghi WFDB)")
    args = ap.parse_args()

    if args.file:
        res = predict_record(args.file)
        print(f"File: {args.file}")
    else:
        # Demo với mẫu có sẵn trong X_ecg.npy
        x_path = HERE / "X_ecg.npy"
        if x_path.exists():
            X = np.load(x_path)
            model = load_model()
            pred = model.predict(X[:5])
            proba = model.predict_proba(X[:5])
            print("Demo 5 mẫu đầu tiên từ X_ecg.npy:")
            for i, (p, pb) in enumerate(zip(pred, proba)):
                lbl = LABEL_MAPPING.get(int(p), "N/A")
                print(f"Mẫu #{i+1}: Dự đoán = {lbl} | Xác suất bất thường = {pb[1]:.4f}")
            return
        else:
            print("Chưa có X_ecg.npy để demo.")
            return

    print(f"Kết quả: {res['label']}")
    print(f"Xác suất bất thường: {res['probability']:.2%}")
    print(f"Độ tin cậy: {res['confidence']:.2%}")
    if res["estimated_hr"]:
        print(f"Nhịp tim ước tính (Lead II): {res['estimated_hr']} bpm")


if __name__ == "__main__":
    main()
