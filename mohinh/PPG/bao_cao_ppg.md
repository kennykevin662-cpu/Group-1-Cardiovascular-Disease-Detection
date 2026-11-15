# Báo cáo kết quả phân hệ PPG

- **Dataset**: PPG-BP Dataset (657 bản ghi PPG thô và hồ sơ lâm sàng), tần số lấy mẫu 1000 Hz.
- **Nhãn mục tiêu**: Nhịp tim HR (bpm) thực tế đo bằng máy đo lâm sàng.
- **Tiền xử lý**: Lọc thông dải Butterworth 0.5–8.0 Hz, chuẩn hóa Z-score.
- **Đặc trưng**: 16 đặc trưng gồm thống kê sóng, biên độ đỉnh, nhịp tim ước tính từ đỉnh tâm thu, RMSSD (HRV).
- **Mô hình**: Random Forest Regressor (300 estimators, min_samples_leaf=2).

## Kết quả trên tập kiểm thử (Test Set)

| Chỉ số đánh giá | Giá trị |
|---|---|
| MAE (Sai số tuyệt đối trung bình) | 3.40 bpm |
| RMSE (Căn bậc hai sai số toàn phương) | 4.78 bpm |
| Hệ số xác định R² | 0.707 |
| MAE Baseline (Đoán nhịp tim trung bình) | 7.38 bpm |
| MAE Baseline (Nhịp tim tính trực tiếp từ đỉnh) | 5.48 bpm |

## Đánh giá & Nhận xét
- Mô hình Random Forest học được tương quan phi tuyến giữa đặc trưng hình thái sóng PPG và nhịp tim thực tế, cải thiện độ chính xác so với việc chỉ bắt đỉnh đơn thuần khi có nhiễu sóng.
