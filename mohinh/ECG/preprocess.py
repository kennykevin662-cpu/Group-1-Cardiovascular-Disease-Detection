"""THÀNH VIÊN 1 – ECG | Dataset: PTB-XL (PhysioNet)
Bước 1-3: tải dữ liệu, khám phá, tiền xử lý
Output: X_ecg.npy, y_ecg.npy (lưu trong mohinh/ECG/)
"""

# TODO 1: Tìm hiểu và tải dataset PTB-XL (đặt vào dataset/ECG/)
def load_dataset(path="dataset/ECG"):
    # TODO: đọc ptbxl_database.csv, scp_statements.csv và các file tín hiệu (wfdb)
    pass


# TODO 2: Khám phá dữ liệu, phân tích nhãn và đặc trưng
def explore_data(df):
    # TODO: thống kê nhãn (NORM, MI, STTC, CD, HYP), phân bố tuổi/giới tính
    # TODO: vẽ tín hiệu ECG mẫu theo từng nhãn
    pass


# TODO 3: Tiền xử lý tín hiệu ECG (lọc nhiễu, chuẩn hóa)
def preprocess_signal(signal, fs=100):
    # TODO: lọc bandpass (0.5 – 40 Hz), loại bỏ baseline wander
    # TODO: chuẩn hóa tín hiệu (z-score / min-max)
    pass


def build_dataset():
    # TODO: chạy load -> explore -> preprocess, chia train/val/test theo strat_fold
    # TODO: lưu X_ecg.npy, y_ecg.npy
    pass


if __name__ == "__main__":
    build_dataset()
