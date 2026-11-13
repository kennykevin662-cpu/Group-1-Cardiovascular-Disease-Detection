"""THÀNH VIÊN 1 – ECG
Hàm dự đoán để Thành viên 4 gọi khi tích hợp vào giao diện."""

import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # để import được preprocess khi gọi từ System/UI
from preprocess import extract_features  # noqa: E402  (đã gọi preprocess_signal bên trong)

_model = None


def _load_model():
    """Load ecg_model.pkl một lần duy nhất rồi dùng lại."""
    global _model
    if _model is None:
        path = HERE / "ecg_model.pkl"
        if not path.exists():
            raise FileNotFoundError(
                "Chưa có ecg_model.pkl. Hãy chạy preprocess.py rồi train.py trước.")
        _model = joblib.load(path)
    return _model


def predict(signal):
    # Nhận đường dẫn record WFDB hoặc mảng numpy
    if isinstance(signal, (str, Path)):
        import wfdb
        signal, _ = wfdb.rdsamp(str(signal))
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2 or signal.shape[1] != 12:
        raise ValueError("ECG phải có dạng (số_mẫu, 12 lead), lấy mẫu 100Hz.")
    if not np.isfinite(signal).all():
        raise ValueError("Tín hiệu có giá trị NaN hoặc vô cực.")

    # Tiền xử lý (bandpass 0.5-40Hz) + trích đặc trưng, giống lúc train
    feats = extract_features(signal).reshape(1, -1)

    model = _load_model()
    prob_abnormal = float(model.predict_proba(feats)[0, 1])
    label = int(prob_abnormal >= 0.5)
    return {
        "label": label,
        "ten_nhan": "Bat thuong" if label else "Binh thuong",
        "xac_suat_bat_thuong": round(prob_abnormal, 4),
        "xac_suat_binh_thuong": round(1 - prob_abnormal, 4),
    }