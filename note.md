# BÁO CÁO KỸ THUẬT VÀ TÀI LIỆU HỆ THỐNG
# HỆ THỐNG AI ĐA PHƯƠNG THỨC PHÁT HIỆN BỆNH TIM MẠCH
*(Multi-modal AI System for Cardiovascular Disease Detection)*

---

## MỤC LỤC
1. [Tổng quan dự án & Ý tưởng thiết kế](#1-tổng-quan-dự-án--ý-tưởng-thiết-kế)
2. [Cơ sở lý thuyết & Phương pháp từng phân hệ](#2-cơ-sở-lý-thuyết--phương-pháp-từng-phân-hệ)
   - 2.1. Phân hệ Dữ liệu Lâm sàng (Clinical) - Thành viên 3
   - 2.2. Phân hệ Điện tâm đồ (ECG) - Thành viên 1
   - 2.3. Phân hệ Quang thể tích (PPG) - Thành viên 2
   - 2.4. Phân hệ Tích hợp Hệ thống & Giao diện (System & UI) - Thành viên 4
3. [Báo cáo sửa lỗi chi tiết (Bug Audit & Fixes)](#3-báo-cáo-sửa-lỗi-chi-tiết-bug-audit--fixes)
   - 3.1. Lỗi phân hệ ECG: Tệp `predict.py` bị ghi đè trùng lặp
   - 3.2. Lỗi phân hệ PPG: Sai lệch chỉ số đặc trưng (Index mismatch)
   - 3.3. Lỗi phân hệ PPG: Bất tương thích dữ liệu & thiếu tệp mô hình
   - 3.4. Lỗi phân hệ Clinical: Lỗi nhập khẩu phụ thuộc chéo và bảng mã console
   - 3.5. Lỗi hạ tầng: Thiếu thư viện phụ thuộc và định dạng `requirements.txt`
4. [Kết quả kiểm thử hệ thống (System Verification & Monitoring)](#4-kết-quả-kiểm-thử-hệ-thống-system-verification--monitoring)
5. [Hướng dẫn vận hành & Khởi chạy giao diện](#5-hướng-dẫn-vận-hành--khởi-chạy-giao-diện)

---

## 1. TỔNG QUAN DỰ ÁN & Ý TƯỞNG THIẾT KẾ

### 1.1. Bối cảnh và Mục tiêu
Bệnh tim mạch (Cardiovascular Diseases - CVDs) là nguyên nhân gây tử vong hàng đầu trên thế giới. Việc phát hiện sớm nguy cơ tim mạch thông qua các dấu hiệu sinh tồn và dữ liệu lâm sàng đóng vai trò then chốt trong điều trị và phòng ngừa rủi ro.

Hệ thống được xây dựng nhằm mục đích:
- Khai thác sức mạnh của **trí tuệ nhân tạo đa phương thức (Multi-modal AI)** bằng cách kết hợp:
  1. Tín hiệu điện tâm đồ đa kênh (**ECG** - Electrocardiogram).
  2. Tín hiệu xung mạch quang học (**PPG** - Photoplethysmogram).
  3. Chỉ số xét nghiệm & bệnh sử lâm sàng (**Clinical** parameters).
- Xây dựng lớp xử lý trung gian (**System**) làm cầu nối chuẩn hóa dữ liệu, xác thực dữ liệu đầu vào, gọi mô hình suy luận, giám sát kiểm thử và tổng hợp báo cáo.
- Cung cấp giao diện trực quan (**UI - Streamlit**) phục vụ bác sĩ, nhân viên y tế hoặc người dùng tự tầm soát nguy cơ tim mạch.

### 1.2. Phân công trách nhiệm nhóm

| Thành viên | Vai trò | Phân hệ / Nhiệm vụ | Nguồn dữ liệu | Thư mục mã nguồn | Trạng thái |
|:---:|:---:|:---|:---|:---|:---:|
| 1 | Model Developer | Huấn luyện mô hình phân tích tín hiệu ECG | PTB-XL (PhysioNet) | `mohinh/ECG/` | Đã hoàn thiện |
| 2 | Model Developer | Huấn luyện mô hình ước lượng nhịp tim & phân tích PPG | PPG-BP / BIDMC | `mohinh/PPG/` | Đã hoàn thiện & sửa lỗi |
| 3 | Model Developer | Huấn luyện mô hình phân loại rủi ro tim mạch lâm sàng | UCI Heart Disease (Cleveland) | `mohinh/Clinical/` | Đã hoàn thiện |
| **4** | **System & UI** | **Thiết kế System, giao diện Streamlit, tích hợp mô hình, kiểm thử & sửa lỗi** | **Toàn bộ hệ thống** | **`System/`, `UI/`, `main.py`** | **Đã hoàn thiện 100%** |

---

## 2. CƠ SỞ LÝ THUYẾT & PHƯƠNG PHÁP TỪNG PHÂN HỆ

### 2.1. Phân hệ Dữ liệu Lâm sàng (Clinical - Thành viên 3)

#### A. Bộ dữ liệu
- **Nguồn**: UCI Machine Learning Repository - Heart Disease Dataset (`processed.cleveland.data`).
- **Quy mô**: 303 bệnh nhân, 14 thuộc tính:
  - *5 biến liên tục (Numeric)*: `age` (tuổi), `trestbps` (huyết áp lúc nghỉ), `chol` (cholesterol máu), `thalach` (nhịp tim tối đa đạt được), `oldpeak` (đoạn ST chênh xuống khi gắng sức).
  - *8 biến phân loại (Categorical)*: `sex` (giới tính), `cp` (loại đau ngực 1-4), `fbs` (đường huyết lúc đói > 120 mg/dl), `restecg` (kết quả điện tâm đồ lúc nghỉ), `exang` (đau thắt ngực do gắng sức), `slope` (độ dốc đoạn ST), `ca` (số mạch máu lớn soi huỳnh quang), `thal` (tình trạng thalassaemia).
  - *Biến mục tiêu (Target)*: Cột `num` (0: Không bệnh, 1-4: Các mức độ bệnh tim) được nhị phân hóa thành:
    $$\text{target} = \begin{cases} 0 & \text{nếu } num = 0 \text{ (Không bệnh)} \\ 1 & \text{nếu } num > 0 \text{ (Có bệnh tim)} \end{cases}$$

#### B. Phương pháp xử lý dữ liệu & Mô hình hóa
1. **Pipeline tiền xử lý (`clinical_scaler.pkl`)**:
   - Biến số: Điền khuyết bằng Trung vị (`SimpleImputer(strategy='median')`) $\to$ Chuẩn hóa Z-score (`StandardScaler`).
   - Biến phân loại: Điền khuyết bằng Yếu vị (`SimpleImputer(strategy='most_frequent')`) $\to$ Mã hóa One-Hot (`OneHotEncoder(handle_unknown='ignore')`).
   - Ghép nối thông qua `ColumnTransformer(sparse_threshold=0)` tạo không gian đặc trưng 27 chiều.
2. **Huấn luyện & Đánh giá**:
   - So sánh đa mô hình: Logistic Regression, Random Forest, Gradient Boosting, XGBoost.
   - Thẩm định chéo 5-fold Stratified K-Fold kết hợp tinh chỉnh siêu tham số qua Grid Search (`scoring='f1'`).
   - Mô hình tối ưu được lưu trữ tại `clinical_model.pkl`.

---

### 2.2. Phân hệ Điện tâm đồ (ECG - Thành viên 1)

#### A. Bộ dữ liệu
- **Nguồn**: PTB-XL Dataset (PhysioNet), bộ dữ liệu 12 chuyển đạo tiêu chuẩn lâm sàng ghi nhận trên 18.885 bệnh nhân, sử dụng bản hạ tần 100 Hz (`records100`).
- **Phân loại nhãn**:
  - Mã chẩn đoán `scp_codes` được ánh xạ về 5 siêu lớp: `NORM` (Bình thường), `MI` (Nhồi máu cơ tim), `STTC` (Biến đổi đoạn ST/T), `CD` (Rối loạn dẫn truyền), `HYP` (Phì đại cơ tim).
  - Nhị phân hóa: $\text{Nhãn } 0$ (Chỉ có NORM - Bình thường), $\text{Nhãn } 1$ (Bất kỳ bất thường tim mạch nào).

#### B. Quy trình xử lý tín hiệu & Trích xuất đặc trưng (89 đặc trưng)
1. **Lọc nhiễu dải thông (Bandpass Filtering)**:
   - Sử dụng bộ lọc số Butterworth bậc 3 (`sosfiltfilt`) dải thông $0.5 - 40.0\text{ Hz}$.
   - Mục đích: Triệt tiêu trôi đường đẳng điện do hô hấp ($< 0.5\text{ Hz}$) và loại bỏ nhiễu cơ học, nhiễu tần số điện lưới ($> 40\text{ Hz}$).
2. **Trích xuất đặc trưng dạng bảng (89 chiều)**:
   - **84 đặc trưng hình thái**: 12 chuyển đạo $\times$ 7 chỉ số thống kê (`mean`, `std`, `min`, `max`, `ptp` - khoảng biến thiên, `skew` - độ lệch, `kurtosis` - độ nhọn).
   - **5 đặc trưng nhịp tim / HRV (Lead II)**: Phát hiện đỉnh sóng R qua thuật toán đỉnh lồi (`scipy.signal.find_peaks` với khoảng cách tối thiểu $0.3\text{s}$ và ngưỡng biên độ phân vị 90):
     - Nhịp tim ước tính: $\text{HR} = \frac{60}{\overline{RR}}$ (bpm).
     - Khoảng cách RR trung bình ($\overline{RR}$) và độ lệch chuẩn ($\text{std}(RR)$).
     - Chỉ số biến thiên nhịp tim RMSSD: $\text{RMSSD} = \sqrt{\frac{1}{N-1}\sum (RR_{i+1} - RR_i)^2}$.
     - Số lượng sóng R phát hiện trong khoảng thời gian 10 giây.
3. **Mô hình học máy**:
   - `RandomForestClassifier` (300 cây, `class_weight='balanced'` để xử lý mất cân bằng dữ liệu, `min_samples_leaf=2`). Mô hình lưu tại `ecg_model.pkl` (kích thước $\approx 6.67\text{ MB}$).

---

### 2.3. Phân hệ Quang thể tích (PPG - Thành viên 2)

#### A. Bộ dữ liệu
- **Nguồn**: Bộ dữ liệu thực tế trong máy là **PPG-BP Dataset** (Guangdong Provincial People's Hospital) gồm:
  - 657 tệp tín hiệu PPG dạng thô (`.txt` tại `Data File/0_subject/`) thu nhận ở tần số 1000 Hz (2100 mẫu / 2.1 giây mỗi tệp).
  - Bảng lâm sàng `PPG-BP dataset.xlsx` chứa Huyết áp tâm thu (SBP), Huyết áp tâm trương (DBP), Nhịp tim (HR), Tăng huyết áp (Hypertension), Đái tháo đường,...

#### B. Phương pháp xử lý tín hiệu & Mô hình hóa
1. **Lọc thông dải (Bandpass Filter 0.5 - 8.0 Hz)**: Giữ lại dải tần số xung nhịp sinh lý tim đập.
2. **Chuẩn hóa Z-score**: Đưa biên độ tín hiệu về giá trị phân chuẩn trung bình 0, độ lệch chuẩn 1.
3. **Trích xuất đặc trưng (16 chiều)**:
   - Thống kê biên độ sóng: Mean, Std, Median, Min, Max, Peak-to-Peak, RMS, Skewness, Kurtosis.
   - Đặc trưng đỉnh xung tâm thu: Số lượng đỉnh (`peak_count`), khoảng cách RR/IBI trung bình, độ lệch chuẩn khoảng cách RR, biên độ đỉnh trung bình.
   - Nhịp tim ước tính từ khoảng cách đỉnh tâm thu (`estimated_hr`).
   - Biến thiên nhịp tim quang học RMSSD (`interval_rmssd`).
   - Khoảng biến thiên biên độ đỉnh (`ptp_peak_amplitudes`).
4. **Mô hình hồi quy**:
   - `RandomForestRegressor` (300 cây, `min_samples_leaf=2`) huấn luyện trên 1.314 đoạn đo, tối ưu hóa dự đoán nhịp tim thực tế từ đặc trưng hình thái sóng PPG.
   - Kết quả đạt **MAE = 3.40 bpm**, **RMSE = 4.78 bpm**, **R² = 0.707**, vượt trội rõ rệt so với baseline đoán trung bình (MAE = 7.38 bpm) và bắt đỉnh đơn thuần (MAE = 5.48 bpm).

---

### 2.4. Phân hệ Tích hợp Hệ thống & Giao diện (System & UI - Thành viên 4)

Kiến trúc phân tầng chặt chẽ:
```
                      ┌─────────────────────────────────────────┐
                      │        GIAO DIỆN STREAMLIT (UI)         │
                      │  - Dashboard tổng quan đa phương thức   │
                      │  - Form nhập thông số lâm sàng (13 chỉ số)│
                      │  - Tải file & trực quan hóa ECG (12 lead)│
                      │  - Tải file & bắt đỉnh xung PPG         │
                      │  - Giám sát & Chạy Unit Test trực quan  │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │              SYSTEM LAYER               │
                      │  - hamhotro.py: Validate, Format, Error │
                      │  - screening.py: Điều phối gọi Model    │
                      │  - dashboard.py: Điểm rủi ro tích hợp   │
                      │  - report.py: Sinh phiếu kết quả y tế   │
                      │  - monitoring.py: 5 bộ kiểm thử tự động │
                      └────────────────────┬────────────────────┘
                                           │
        ┌──────────────────────────────────┼──────────────────────────────────┐
        ▼                                  ▼                                  ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│   MÔ HÌNH ECG    │               │   MÔ HÌNH PPG    │               │ MÔ HÌNH CLINICAL │
│  ecg_model.pkl   │               │  ppg_model.pkl   │               │clinical_model.pkl│
│ (Random Forest)  │               │ (Random Forest)  │               │(GradientBoosting)│
└──────────────────┘               └──────────────────┘               └──────────────────┘
```

1. **`System/hamhotro.py`**:
   - `validate_input(data, input_type)`: Kiểm tra cấu trúc dữ liệu, kiểm tra dữ liệu rỗng và xác minh giá trị sinh học (ví dụ: tuổi 1-120, huyết áp 50-260 mmHg, nhịp tim 40-240 bpm).
   - `validate_file(file_path, allowed_exts)`: Kiểm tra tệp tồn tại, kích thước > 0 và phần mở rộng cho phép (.csv, .npy, .txt).
   - `format_prediction(res, modality)`: Chuẩn hóa kết quả suy luận ra định dạng JSON/dict đồng nhất.
   - `handle_error(e, context)`: Bắt ngoại lệ và sinh câu giải thích thân thiện với y tế.
2. **`System/screening.py`**:
   - Cung cấp 3 API giao tiếp: `screen_clinical()`, `screen_ecg()`, `screen_ppg()`.
3. **`System/dashboard.py`**:
   - Thuật toán tính **Điểm rủi ro tim mạch toàn diện (Cardiovascular Risk Index - CRI)**:
     $$\text{CRI} = 0.40 \times \text{Score}_{\text{Clinical}} + 0.40 \times \text{Score}_{\text{ECG}} + 0.20 \times \text{Score}_{\text{PPG}}$$
   - Phân loại 4 cấp độ: Thấp (<30%), Trung bình (30-49%), Cao (50-69%), Rất cao (≥70%).
4. **`System/report.py`**:
   - Tạo phiếu chẩn đoán y khoa hoàn chỉnh dưới định dạng Markdown, hỗ trợ xem trước và tải về.
5. **`System/monitoring.py`**:
   - Chạy 5 ca kiểm thử tự động kiểm tra ECG, PPG, Clinical, Input ngoại lệ và luồng End-to-End.

---

## 3. BÁO CÁO SỬA LỖI CHI TIẾT (BUG AUDIT & FIXES)

Dưới đây là bảng tổng hợp tất cả các lỗi được phát hiện trong dự án và chi tiết cách Thành viên 4 đã khắc phục:

### 3.1. Lỗi phân hệ ECG: Tệp `predict.py` bị ghi đè trùng lặp
- **Vị trí**: `mohinh/ECG/predict.py`
- **Mức độ**: 🔴 Blocker (Nghiêm trọng nhất)
- **Mô tả lỗi**: Tệp `predict.py` có kích thước 7.290 bytes, nội dung giống hệt 100% tệp `preprocess.py`. Tệp này không hề có hàm `predict()` mà lại chứa hàm `build_dataset()`. Khi System gọi tệp này, chương trình sẽ không thể dự đoán mà ngược lại cố gắng đọc và lọc lại toàn bộ dataset PTB-XL.
- **Lý do**: Thành viên 1 vô tình copy nhầm nội dung `preprocess.py` đè lên `predict.py` trước khi bàn giao.
- **Cách sửa (Highlight)**:
  - Viết lại hoàn chỉnh `mohinh/ECG/predict.py`:
    + Hàm `load_model()`: Tải `ecg_model.pkl`.
    + Hàm `predict(signal, fs=100)`: Tự động điều chỉnh số kênh (tile thành 12 chuyển đạo nếu nhận tín hiệu đơn kênh), trích xuất 89 đặc trưng qua `extract_features`, gọi `model.predict` và `model.predict_proba`.
    + Hàm `predict_record(record_path)`: Hỗ trợ nạp trực tiếp file `.npy`, `.csv` hoặc bản ghi WFDB.
    + Thêm khối `try...except` nạp module linh hoạt bất kể ngữ cảnh thực thi.
- **Kết quả sau sửa**: Hàm `predict` chạy ổn định, trả về kết quả dự đoán (Bình thường / Bất thường), xác suất và nhịp tim Lead II.

---

### 3.2. Lỗi phân hệ PPG: Sai lệch chỉ số đặc trưng (Index Mismatch)
- **Vị trí**: `mohinh/PPG/predict.py` (dòng 21) và `mohinh/PPG/train.py` (dòng 37)
- **Mức độ**: 🔴 Nghiêm trọng (Logic Bug)
- **Mô tả lỗi**:
  ```python
  # TRƯỚC KHI SỬA (CODE CỦA THÀNH VIÊN 2):
  # Trong predict.py dòng 21:
  hr_peaks = float(feats[0])  # LẤY NHẦM CHỈ SỐ 0

  # Trong train.py dòng 37:
  mae_peaks = mean_absolute_error(y_test, X_test[:, 0])  # LẤY NHẦM CỘT 0
  ```
- **Lý do**: Trong `mohinh/PPG/preprocess.py`, danh sách đặc trưng được sắp xếp như sau:
  - `features[0] = np.mean(x)` $\to$ Do tín hiệu đã qua chuẩn hóa Z-score nên trung bình `mean(x)` luôn xấp xỉ bằng **0.0**.
  - `features[13] = estimated_hr` $\to$ Đây mới thực sự là nhịp tim tính từ khoảng cách đỉnh tâm thu!
  - Thành viên 2 đã nhớ nhầm chỉ số, dẫn đến việc lấy giá trị trung bình biên độ ($0.0$) gán làm nhịp tim ($0\text{ bpm}$).
- **Cách sửa (Highlight)**:
  ```python
  # SAU KHI SỬA (THÀNH VIÊN 4 ĐÃ SỬA):
  # Trong predict.py:
  hr_peaks = float(feats[13])  # ĐÚNG CHỈ SỐ 13: ESTIMATED_HR TỪ ĐỈNH
  rmssd = float(feats[14])     # ĐÚNG CHỈ SỐ 14: RMSSD (HRV)

  # Trong train.py:
  mae_peaks = mean_absolute_error(y_test, X_test[:, 13])  # ĐÚNG CỘT 13
  ```
- **Kết quả sau sửa**: Nhịp tim đo từ đỉnh xung hiển thị chính xác (ví dụ tệp mẫu `100_1.txt` cho ra **74.9 bpm**, khớp hoàn hảo với nhịp tim thực tế **73 bpm** của bệnh nhân).

---

### 3.3. Lỗi phân hệ PPG: Bất tương thích dữ liệu & thiếu tệp mô hình
- **Vị trí**: `mohinh/PPG/preprocess.py` và thư mục `dataset/PPG/`
- **Mức độ**: 🔴 Blocker
- **Mô tả lỗi**:
  - `README.md` và mã nguồn cũ của TV2 viết để đọc tệp CSV của PhysioNet BIDMC (`*Signals.csv`).
  - Dữ liệu thực tế có trong `dataset/PPG/` lại là **PPG-BP Dataset** (gồm 657 tệp `.txt` trong `0_subject/` và `PPG-BP dataset.xlsx`).
  - Hậu quả: Hàm `build_dataset()` văng lỗi `FileNotFoundError`, thư mục `mohinh/PPG/` hoàn toàn **chưa có mô hình `ppg_model.pkl`** và tệp `X_ppg.npy`.
- **Cách sửa (Highlight)**:
  - Cập nhật `mohinh/PPG/preprocess.py`: Xây dựng hàm `load_dataset()` thông minh, tự động nhận diện cả tệp Excel `PPG-BP dataset.xlsx` và nạp 1.314 mẫu đo từ các tệp `.txt`.
  - Cấu hình tần số lấy mẫu chuẩn cho PPG-BP: `fs = 1000 Hz`.
  - Chạy trích xuất đặc trưng và lưu thành công `X_ppg.npy` (1.314 mẫu, 16 đặc trưng) và `y_ppg.npy`.
  - Chạy huấn luyện mô hình `RandomForestRegressor` trong `train.py` và lưu trữ tệp mô hình chuẩn `ppg_model.pkl`.
- **Kết quả sau sửa**: Mô hình PPG hoàn thiện, dự đoán nhịp tim đạt sai số cực thấp $MAE = 3.40\text{ bpm}$.

---

### 3.4. Lỗi phân hệ Clinical: Lỗi nhập khẩu phụ thuộc chéo và bảng mã console
- **Vị trí**: `mohinh/Clinical/predict.py` (dòng 14-19)
- **Mức độ**: 🟡 Trung bình
- **Mô tả lỗi**:
  1. `predict.py` thực hiện lệnh `from train import MODEL_PATH`. Khi `predict.py` được import từ thư mục khác (như `System/`), Python cố import `train.py`, nhưng trong `train.py` dòng 15 lại gọi `from preprocess import ...` mà không có prefix package, gây ra ngoại lệ `ModuleNotFoundError: No module named 'preprocess'`.
  2. In chuỗi tiếng Việt có dấu trên console Windows CP1252 gây lỗi `UnicodeEncodeError`.
- **Cách sửa (Highlight)**:
  - Trong `predict.py`, tự định nghĩa trực tiếp `MODEL_PATH = HERE / "clinical_model.pkl"` và `SCALER_PATH = HERE / "clinical_scaler.pkl"` thay vì import từ `train.py`.
  - Bổ sung hàm `predict_dict(data)` hỗ trợ System truyền trực tiếp dictionary 13 trường.
  - Cấu hình `sys.stdout.reconfigure(encoding='utf-8')` xử lý triệt để lỗi hiển thị tiếng Việt trên Windows.
- **Kết quả sau sửa**: Module Clinical được import và gọi suy luận trơn tru từ mọi phân hệ.

---

### 3.5. Lỗi hạ tầng: Thiếu thư viện phụ thuộc và định dạng `requirements.txt`
- **Vị trí**: `requirements.txt`
- **Mức độ**: 🟠 Nghiêm trọng
- **Mô tả lỗi**: File `requirements.txt` được lưu bằng bảng mã UTF-16LE gây khó khăn khi đọc tự động, đồng thời thiếu mất hai thư viện bắt buộc để hệ thống vận hành: `streamlit` (giao diện UI) và `openpyxl` (đọc bảng dữ liệu PPG-BP).
- **Cách sửa (Highlight)**:
  - Chuyển mã `requirements.txt` về chuẩn UTF-8.
  - Bổ sung `streamlit>=1.67.0` và `openpyxl>=3.1.5`.
  - Cài đặt thành công `streamlit 1.67.0`, `openpyxl 3.1.5`, `wfdb 4.3.1`, `tqdm 4.70.1` vào môi trường Python.

---

## 4. KẾT QUẢ KIỂM THỬ HỆ THỐNG (SYSTEM VERIFICATION & MONITORING)

Toàn bộ 5 bộ kiểm thử tự động trong `System/monitoring.py` đã được thực thi và vượt qua với tỷ lệ thành công tuyệt đối **100% (5/5)**:

```
=================================================================
 BẮT ĐẦU CHẠY KIỂM THỬ HỆ THỐNG (SYSTEM MONITORING)
=================================================================
[TEST 1/5] Kiểm thử mô hình ECG...
  ✓ Thành công: Label='Bất thường (Cardiovascular Abnormality)', Xác suất=0.8707
[TEST 2/5] Kiểm thử mô hình PPG...
  ✓ Thành công: Nhịp tim dự đoán=69.8 bpm, Trạng thái='Nhịp tim bình thường (Normal Sinus Rhythm)'
[TEST 3/5] Kiểm thử mô hình Clinical...
  ✓ Thành công: Label='Không bệnh', Xác suất có bệnh=0.0878
[TEST 4/5] Kiểm thử bắt lỗi đầu vào bất hợp lệ...
  ✓ Thành công: Cả 4 trường hợp ngoại lệ đều được phát hiện và ngăn chặn chính xác.
[TEST 5/5] Kiểm thử luồng tích hợp toàn hệ thống (End-to-End)...
  ✓ Thành công: Luồng tích hợp End-to-End hoàn chỉnh (Điểm rủi ro tổng thể: 41.4/100 - Nguy cơ trung bình).
=================================================================
 KẾT QUẢ: ĐÃ VƯỢT QUA 5/5 BỘ KIỂM THỬ (100%)
=================================================================
```

---

## 5. HƯỚNG DẪN VẬN HÀNH & KHỞI CHẠY GIAO DIỆN

### 5.1. Khởi chạy giao diện web Streamlit (Khuyên dùng)
Từ thư mục gốc dự án, mở terminal/powershell và chạy lệnh:
```bash
streamlit run UI/app.py
```
Hoặc khởi chạy thông qua tệp `main.py`:
```bash
python main.py --run
```
Giao diện ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

### 5.2. Chạy bộ kiểm thử tự động toàn diện
```bash
python main.py --test
```

### 5.3. Sử dụng các chức năng trên giao diện
1. **Thanh bên (Sidebar)**: Nhập thông tin hồ sơ bệnh nhân (Họ tên, Mã bệnh nhân, Tuổi, Giới tính) và chọn phân hệ làm việc.
2. **Tab Sàng lọc Lâm sàng (`Clinical`)**: Nhập 13 trường chỉ số hoặc nhấn các nút *"Nạp mẫu Nguy cơ thấp"* / *"Nạp mẫu Nguy cơ cao"* để thử nghiệm ngay kết quả dự đoán.
3. **Tab Sàng lọc Điện tâm đồ (`ECG`)**: Chọn mẫu đo từ PTB-XL hoặc tải tệp `.npy`, chọn kênh (Lead I, II, V1...) để quan sát dạng sóng và nhấn nút phân tích.
4. **Tab Sàng lọc Quang thể tích (`PPG`)**: Chọn tệp mẫu thực tế trong `0_subject/` hoặc tải file thô, quan sát các đỉnh xung tâm thu được đánh dấu tự động và xem nhịp tim ước lượng.
5. **Tab Bảng điều khiển (`Dashboard`)**: Theo dõi điểm rủi ro tim mạch tổng hợp CRI (0-100), xem khuyến nghị của bác sĩ AI và nhấn nút **Tải Phiếu Báo cáo (.md)** để lưu hồ sơ bệnh án.
