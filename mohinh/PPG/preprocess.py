"""THÀNH VIÊN 2 – PPG | Dataset: BIDMC PPG and Respiration (PhysioNet)
Bước 1-4: tải dữ liệu, khám phá, tiền xử lý, trích đặc trưng
Output: X_ppg.npy, y_ppg.npy (lưu trong mohinh/PPG/)
"""

# TODO 1: Tìm hiểu và tải dataset BIDMC PPG and Respiration (đặt vào dataset/PPG/)
def load_dataset(path="dataset/PPG"):
    # TODO: đọc các file tín hiệu (wfdb) và số liệu (HR, RR, SpO2...)
    pass


# TODO 2: Khám phá và trực quan hóa tín hiệu PPG
def explore_data(data):
    # TODO: vẽ tín hiệu PPG theo thời gian, phổ tần số
    # TODO: thống kê độ dài, tần số lấy mẫu, phân bố nhãn
    pass


# TODO 3: Tiền xử lý tín hiệu, lọc nhiễu
def preprocess_signal(signal, fs=125):
    # TODO: lọc bandpass (0.5 – 8 Hz), loại bỏ nhiễu chuyển động, chuẩn hóa
    pass


# TODO 4: Trích xuất đặc trưng như nhịp mạch, khoảng cách giữa các đỉnh
def extract_features(signal, fs=125):
    # TODO: phát hiện đỉnh (scipy.signal.find_peaks)
    # TODO: tính heart rate, khoảng cách peak-to-peak, HRV, biên độ...
    pass


def build_dataset():
    # TODO: chạy load -> explore -> preprocess -> extract_features
    # TODO: lưu X_ppg.npy, y_ppg.npy
    pass


if __name__ == "__main__":
    build_dataset()
