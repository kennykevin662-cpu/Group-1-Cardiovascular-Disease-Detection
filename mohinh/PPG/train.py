"""THÀNH VIÊN 2 – PPG
Bước 5-6: xây dựng mô hình, đánh giá, lưu
Output: ppg_model.pkl (và ppg_scaler.pkl nếu có)
"""
import os
 
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
 
DATA_DIR = "mohinh/PPG"
 
 
# TODO 5: Xây dựng mô hình phân tích tín hiệu phù hợp với nhãn dữ liệu
def build_model():
    # TODO: chọn mô hình (Regression / Random Forest / CNN...) phù hợp nhãn
    # Nhãn là nhịp tim (bpm, số thực) -> hồi quy -> Random Forest Regressor
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
 
def train(model, X_train, y_train):
    # TODO: huấn luyện mô hình
    model.fit(X_train, y_train)
    return model
 
 
# TODO 6: Đánh giá kết quả và viết báo cáo phần PPG
def evaluate(model, X_test, y_test):
    # TODO: chỉ số phù hợp (MAE/RMSE nếu hồi quy; Accuracy/F1 nếu phân loại)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    # baseline: luôn đoán giá trị trung bình, và HR tính trực tiếp từ đỉnh (cột 0 của X)
    mae_mean = mean_absolute_error(y_test, np.full_like(y_test, np.mean(y_test)))
    mae_peaks = mean_absolute_error(y_test, X_test[:, 0])
    print(f"MAE={mae:.2f} bpm | RMSE={rmse:.2f} bpm | R2={r2:.3f}")
    print(f"Baseline MAE: đoán trung bình={mae_mean:.2f}, HR từ đỉnh={mae_peaks:.2f}")
 
    # TODO: viết báo cáo phần PPG
    report = f"""# Báo cáo phần PPG
 
- Dataset: BIDMC PPG and Respiration (PhysioNet), PPG lấy mẫu 125 Hz.
- Nhãn: nhịp tim HR (bpm) trung bình trong mỗi cửa sổ 10 s.
- Tiền xử lý: lọc bandpass 0.5-8 Hz, chuẩn hóa; trích đặc trưng từ đỉnh, HRV, phổ.
- Mô hình: Random Forest Regressor (bài toán hồi quy).
 
## Kết quả trên tập test
 
| Chỉ số | Giá trị |
|---|---|
| MAE (bpm) | {mae:.2f} |
| RMSE (bpm) | {rmse:.2f} |
| R2 | {r2:.3f} |
| MAE baseline (đoán trung bình) | {mae_mean:.2f} |
| MAE baseline (HR từ đỉnh) | {mae_peaks:.2f} |
 
## Nhận xét
(Bổ sung: so sánh với baseline, các trường hợp sai lớn, hướng cải thiện.)
"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "bao_cao_ppg.md"), "w", encoding="utf-8") as f:
        f.write(report)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
 
def save_model(model, path="mohinh/PPG/ppg_model.pkl"):
    # TODO: lưu mô hình
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print("Đã lưu mô hình:", path)
 
 
if __name__ == "__main__":
    # TODO: load X_ppg.npy, y_ppg.npy -> build -> train -> evaluate -> save
    X = np.load(os.path.join(DATA_DIR, "X_ppg.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_ppg.npy"))
    gpath = os.path.join(DATA_DIR, "groups_ppg.npy")
    if os.path.exists(gpath):  # chia theo bệnh nhân để tránh rò rỉ dữ liệu
        groups = np.load(gpath)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
        X_train, X_test, y_train, y_test = X[tr], X[te], y[tr], y[te]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    model = build_model()
    model = train(model, X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model)
