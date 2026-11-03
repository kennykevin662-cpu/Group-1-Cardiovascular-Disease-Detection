# Group-1-Cardiovascular-Disease-Detection

Hệ thống AI đa phương thức phát hiện bệnh tim mạch (Multi-modal AI System for Cardiovascular Disease Detection)

Dự án phân tích 3 nguồn dữ liệu: tín hiệu điện tâm đồ (ECG), tín hiệu quang thể tích (PPG) và dữ liệu lâm sàng,
tích hợp vào giao diện Streamlit.

## Phân công nhóm

| Thành viên | Phân hệ | Dataset | Thư mục |
|---|---|---|---|
| 1 | ECG | PTB-XL (PhysioNet) | `mohinh/ECG/` |
| 2 | PPG | BIDMC PPG and Respiration (PhysioNet) | `mohinh/PPG/` |
| 3 | Clinical | UCI Heart Disease | `mohinh/Clinical/` |
| 4 | System & UI | – | `System/`, `UI/` |

## Cấu trúc

```
dataset/    dữ liệu (Clinical, ECG, PPG)
mohinh/     mô hình của từng thành viên (Clinical, ECG, PPG)
            mỗi thư mục gồm: preprocess.py -> train.py -> predict.py
System/     tích hợp mô hình
UI/         giao diện Streamlit
main.py
```

## Chạy giao diện

```
streamlit run UI/app.py
```

## Quy trình mỗi phân hệ

```
python mohinh/ECG/preprocess.py   # tạo X_ecg.npy, y_ecg.npy
python mohinh/ECG/train.py        # huấn luyện, đánh giá, lưu ecg_model.pkl
# predict.py được System/UI gọi khi chạy
```
(PPG và Clinical làm tương tự)
