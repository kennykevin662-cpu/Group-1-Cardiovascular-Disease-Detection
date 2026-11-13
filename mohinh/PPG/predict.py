"""THÀNH VIÊN 2 – PPG
Hàm dự đoán để Thành viên 4 gọi khi tích hợp vào giao diện."""
import joblib
import numpy as np
 
from preprocess import preprocess_signal, extract_features
 
FS = 125
 
 
def predict(signal):
    # TODO: load ppg_model.pkl, tiền xử lý, trích đặc trưng
    model = joblib.load("mohinh/PPG/ppg_model.pkl")
    x = preprocess_signal(np.asarray(signal, dtype=float), FS)
    feats = extract_features(x, FS)
    if feats is None:
        return {"heart_rate": None, "message": "Không phát hiện được nhịp mạch (tín hiệu quá ngắn hoặc nhiễu)"}
 
    # TODO: trả về kết quả (nhịp mạch, nhãn/giá trị dự đoán)
    hr_model = float(model.predict(feats.reshape(1, -1))[0])  # nhịp tim do mô hình dự đoán (bpm)
    hr_peaks = float(feats[0])                                # nhịp tim tính từ khoảng cách đỉnh (bpm)
    return {"heart_rate": round(hr_model, 1), "heart_rate_peaks": round(hr_peaks, 1)}
