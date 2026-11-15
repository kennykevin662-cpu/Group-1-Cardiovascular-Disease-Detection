"""THÀNH VIÊN 2 – PPG
Bước 5-6: xây dựng mô hình, đánh giá, lưu
Output: ppg_model.pkl, bao_cao_ppg.md
Đã được sửa lỗi và hoàn thiện bởi Thành viên 4.
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# Đảm bảo console Windows không bị UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE


# TODO 5: Xây dựng mô hình phân tích tín hiệu phù hợp với nhãn dữ liệu
def build_model():
    # Nhãn là nhịp tim (bpm, số thực) -> bài toán hồi quy -> RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )


def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


# TODO 6: Đánh giá kết quả và viết báo cáo phần PPG
def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)

    # Baseline: đoán giá trị trung bình, và HR tính từ đỉnh xung (cột 13 của X_test)
    # LƯU Ý SỬA LỖI: Cột 13 là estimated_hr từ khoảng cách đỉnh, không phải cột 0 (mean(x))
    mae_mean = mean_absolute_error(y_test, np.full_like(y_test, np.mean(y_test)))
    mae_peaks = mean_absolute_error(y_test, X_test[:, 13])

    print(f"MAE={mae:.2f} bpm | RMSE={rmse:.2f} bpm | R2={r2:.3f}")
    print(f"Baseline MAE: đoán trung bình={mae_mean:.2f}, HR từ đỉnh={mae_peaks:.2f}")

    report = f"""# Báo cáo kết quả phân hệ PPG

- **Dataset**: PPG-BP Dataset (657 bản ghi PPG thô và hồ sơ lâm sàng), tần số lấy mẫu 1000 Hz.
- **Nhãn mục tiêu**: Nhịp tim HR (bpm) thực tế đo bằng máy đo lâm sàng.
- **Tiền xử lý**: Lọc thông dải Butterworth 0.5–8.0 Hz, chuẩn hóa Z-score.
- **Đặc trưng**: 16 đặc trưng gồm thống kê sóng, biên độ đỉnh, nhịp tim ước tính từ đỉnh tâm thu, RMSSD (HRV).
- **Mô hình**: Random Forest Regressor (300 estimators, min_samples_leaf=2).

## Kết quả trên tập kiểm thử (Test Set)

| Chỉ số đánh giá | Giá trị |
|---|---|
| MAE (Sai số tuyệt đối trung bình) | {mae:.2f} bpm |
| RMSE (Căn bậc hai sai số toàn phương) | {rmse:.2f} bpm |
| Hệ số xác định R² | {r2:.3f} |
| MAE Baseline (Đoán nhịp tim trung bình) | {mae_mean:.2f} bpm |
| MAE Baseline (Nhịp tim tính trực tiếp từ đỉnh) | {mae_peaks:.2f} bpm |

## Đánh giá & Nhận xét
- Mô hình Random Forest học được tương quan phi tuyến giữa đặc trưng hình thái sóng PPG và nhịp tim thực tế, cải thiện độ chính xác so với việc chỉ bắt đỉnh đơn thuần khi có nhiễu sóng.
"""
    with open(DATA_DIR / "bao_cao_ppg.md", "w", encoding="utf-8") as f:
        f.write(report)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def save_model(model, path=DATA_DIR / "ppg_model.pkl"):
    joblib.dump(model, path)
    print("Đã lưu mô hình:", path)


if __name__ == "__main__":
    X_path = DATA_DIR / "X_ppg.npy"
    y_path = DATA_DIR / "y_ppg.npy"
    g_path = DATA_DIR / "groups_ppg.npy"

    if not (X_path.exists() and y_path.exists()):
        print("Chưa có X_ppg.npy / y_ppg.npy. Đang chạy preprocess...")
        from preprocess import build_dataset
        X, y, groups = build_dataset()
    else:
        X = np.load(X_path)
        y = np.load(y_path)
        groups = np.load(g_path) if g_path.exists() else None

    # Chia theo nhóm bệnh nhân (GroupShuffleSplit) để tránh rò rỉ dữ liệu giữa các đoạn cùng người
    if groups is not None:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
        X_train, X_test, y_train, y_test = X[tr], X[te], y[tr], y[te]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Kích thước tập huấn luyện: {X_train.shape}, tập kiểm thử: {X_test.shape}")
    model = build_model()
    model = train(model, X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model)
