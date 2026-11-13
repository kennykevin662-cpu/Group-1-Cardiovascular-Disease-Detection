"""
THÀNH VIÊN 4 – SYSTEM
Dashboard – Tổng hợp kết quả phân tích ECG, PPG và Clinical.
"""

# ============================================================
# TODO 1: QUẢN LÝ KẾT QUẢ ECG
# ============================================================

# TODO:
# - Nhận kết quả dự đoán từ model ECG.
# - Lưu thông tin prediction.
# - Lưu label/kết luận.
# - Lưu probability/confidence nếu model có.


def get_ecg_result(result):
    # TODO: xử lý kết quả ECG để hiển thị trên UI.
    pass


# ============================================================
# TODO 2: QUẢN LÝ KẾT QUẢ PPG
# ============================================================

# TODO:
# - Nhận kết quả dự đoán từ model PPG.
# - Lưu prediction.
# - Lưu label/kết luận.
# - Lưu probability/confidence nếu có.
# - Lưu các đặc trưng như heart rate nếu có.


def get_ppg_result(result):
    # TODO: xử lý kết quả PPG để hiển thị trên UI.
    pass


# ============================================================
# TODO 3: QUẢN LÝ KẾT QUẢ CLINICAL
# ============================================================

# TODO:
# - Nhận kết quả dự đoán từ model Clinical.
# - Lưu prediction.
# - Lưu label/kết luận.
# - Lưu probability/confidence nếu có.


def get_clinical_result(result):
    # TODO: xử lý kết quả Clinical để hiển thị trên UI.
    pass


# ============================================================
# TODO 4: TỔNG HỢP KẾT QUẢ
# ============================================================

# TODO:
# - Tổng hợp kết quả ECG.
# - Tổng hợp kết quả PPG.
# - Tổng hợp kết quả Clinical.
# - Chuẩn bị dữ liệu cho dashboard.


def get_dashboard_data(ecg=None, ppg=None, clinical=None):
    # TODO: tổng hợp kết quả của 3 mô hình.
    pass