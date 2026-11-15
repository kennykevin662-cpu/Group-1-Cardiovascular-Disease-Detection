"""THÀNH VIÊN 2 – PPG
Hàm dự đoán để Thành viên 4 gọi khi tích hợp vào giao diện và System.
Đã được sửa lỗi và hoàn thiện bởi Thành viên 4.
"""
from pathlib import Path
import joblib
import numpy as np

try:
    from preprocess import DEFAULT_FS, extract_features, preprocess_signal
except ImportError:
    from mohinh.PPG.preprocess import DEFAULT_FS, extract_features, preprocess_signal

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "ppg_model.pkl"


def load_model(path=MODEL_PATH):
    p = Path(path)
    if not p.is_absolute():
        p = HERE / p
    if not p.exists():
        raise FileNotFoundError(f"Chưa có mô hình PPG tại {p}. Hãy chạy train.py trước.")
    return joblib.load(p)


def predict(signal, fs=DEFAULT_FS, model=None):
    """Dự đoán nhịp tim và đánh giá chỉ số từ tín hiệu PPG thô.
    
    Args:
        signal: mảng 1D chứa các mẫu tín hiệu quang thể tích PPG.
        fs: tần số lấy mẫu (mặc định 1000 Hz theo PPG-BP).
        model: mô hình đã load trước đó (tùy chọn).
    
    Returns:
        dict: kết quả phân tích gồm heart_rate (dự đoán của mô hình), 
              heart_rate_peaks (tính từ đỉnh xung), rmssd, status, features.
    """
    if model is None:
        model = load_model()

    arr = np.asarray(signal, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {"status": "error", "message": "Tín hiệu PPG rỗng"}

    clean_x = preprocess_signal(arr, fs=fs)
    feats = extract_features(clean_x, fs=fs)

    # ĐẶC TRƯNG CHÍNH ĐÃ ĐƯỢC SỬA ĐÚNG CHỈ SỐ:
    # feats[13]: estimated_hr tính từ khoảng cách các đỉnh xung
    # feats[14]: interval_rmssd (HRV)
    # feats[9]: số lượng đỉnh phát hiện
    hr_peaks = float(feats[13])
    rmssd = float(feats[14])
    peak_count = int(feats[9])

    # Dự đoán bằng mô hình Random Forest Regressor
    hr_model = float(model.predict(feats.reshape(1, -1))[0])

    # Đánh giá trạng thái nhịp tim
    # Chuẩn sinh lý: 60 - 100 bpm là bình thường lúc nghỉ
    if hr_model < 60:
        status_label = "Nhịp tim chậm (Bradycardia)"
        risk_level = "Cảnh báo"
    elif hr_model > 100:
        status_label = "Nhịp tim nhanh (Tachycardia)"
        risk_level = "Cảnh báo"
    else:
        status_label = "Nhịp tim bình thường (Normal Sinus Rhythm)"
        risk_level = "Bình thường"

    return {
        "status": "success",
        "heart_rate": round(hr_model, 1),
        "heart_rate_peaks": round(hr_peaks, 1) if hr_peaks > 0 else round(hr_model, 1),
        "rmssd": round(rmssd, 4) if rmssd > 0 else None,
        "peak_count": peak_count,
        "label": status_label,
        "risk_level": risk_level,
        "features": feats,
    }


def predict_file(filepath, fs=DEFAULT_FS, model=None):
    """Đọc tệp .txt / .csv / .npy và chạy dự đoán."""
    p = Path(filepath)
    if p.suffix == ".npy":
        sig = np.load(p)
    elif p.suffix == ".txt":
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            tokens = f.read().split()
        sig = np.array([float(x) for x in tokens if x], dtype=np.float32)
    elif p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        ppg_col = [c for c in df.columns if any(k in str(c).lower() for k in ("ppg", "pleth"))]
        col = ppg_col[0] if ppg_col else df.columns[0]
        sig = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
    else:
        raise ValueError(f"Định dạng tệp không được hỗ trợ: {p.suffix}")

    res = predict(sig, fs=fs, model=model)
    res["file_path"] = str(p)
    return res


if __name__ == "__main__":
    import sys
    ROOT = HERE.parents[1]
    sample_files = list((ROOT / "dataset/PPG/Data File/0_subject").glob("*.txt"))
    if sample_files:
        print(f"Kiểm thử với tệp mẫu: {sample_files[0].name}")
        out = predict_file(sample_files[0])
        print(f"Nhịp tim dự đoán (Mô hình): {out['heart_rate']} bpm")
        print(f"Nhịp tim từ đỉnh xung: {out['heart_rate_peaks']} bpm")
        print(f"Trạng thái: {out['label']} ({out['risk_level']})")
    else:
        print("Không tìm thấy tệp .txt mẫu trong dataset/PPG.")
