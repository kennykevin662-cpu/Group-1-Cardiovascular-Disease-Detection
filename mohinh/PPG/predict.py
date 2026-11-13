""""""THÀNH VIÊN 2 – PPG
Hàm dự đoán để Thành viên 4 gọi khi tích hợp vào giao diện."""
import os
 
import joblib
import numpy as np
 
from preprocess import FS, STEP_SEC, WINDOW_SEC, extract_features, preprocess_signal
 
MODEL_PATH = "mohinh/PPG/ppg_model.pkl"
_model = None
 
 
def _get_model(path=None):
    global _model
    path = path or MODEL_PATH
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chưa có mô hình tại '{path}'. Hãy chạy train.py trước.")
        _model = joblib.load(path)
    return _model
 
 
def predict(signal, fs=FS):
    """
    signal: mảng 1D tín hiệu PPG thô (ít nhất ~5 giây).
    Trả về dict:
      - heart_rate: nhịp tim dự đoán bởi mô hình (bpm)
      - heart_rate_peaks: nhịp tim đo trực tiếp từ khoảng cách đỉnh (bpm)
      - n_windows: số cửa sổ dùng để dự đoán
      - status / message: "ok" hoặc "error" kèm lý do (để giao diện hiển thị)
    """
    signal = np.asarray(signal, dtype=float).ravel()
    if len(signal) < 5 * fs:
        return {"status": "error", "message": "Tín hiệu quá ngắn (cần >= 5 giây)."}
 
    x = preprocess_signal(signal, fs)
    win, step = WINDOW_SEC * fs, STEP_SEC * fs
    if len(x) <= win:
        segments = [x]
    else:
        segments = [x[s:s + win] for s in range(0, len(x) - win + 1, step)]
 
    feats = [f for f in (extract_features(s, fs) for s in segments) if f is not None]
    if not feats:
        return {"status": "error", "message": "Không phát hiện được đỉnh mạch, tín hiệu có thể bị nhiễu."}
 
    X = np.vstack(feats)
    preds = _get_model().predict(X)
    return {
        "status": "ok",
        "heart_rate": float(np.median(preds)),
        "heart_rate_peaks": float(np.median(X[:, 0])),
        "n_windows": int(len(feats)),
    }
 
 
if __name__ == "__main__":
    # Chạy thử với PPG giả lập 75 bpm
    t = np.arange(0, 20, 1 / FS)
    demo = np.sin(2 * np.pi * 1.25 * t) + 0.3 * np.sin(2 * np.pi * 2.5 * t) + 0.05 * np.random.randn(len(t))
    print(predict(demo))
