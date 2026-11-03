"""THÀNH VIÊN 3 – Clinical | Dataset: UCI Heart Disease
Bước 1-3: tải dữ liệu, khám phá, xử lý dữ liệu
Output: X_clinical.npy, y_clinical.npy, clinical_scaler.pkl (lưu trong mohinh/Clinical/)
"""

# TODO 1: Tìm hiểu và tải dataset Heart Disease (đặt vào dataset/Clinical/)
def load_dataset(path="dataset/Clinical"):
    # TODO: đọc file csv bằng pandas
    pass


# TODO 2: Khám phá và phân tích dữ liệu
def explore_data(df):
    # TODO: df.info(), describe(), phân bố nhãn target
    # TODO: heatmap tương quan, histogram các đặc trưng (age, chol, thalach...)
    pass


# TODO 3: Xử lý dữ liệu thiếu, mã hóa và chuẩn hóa dữ liệu
def preprocess(df):
    # TODO: xử lý missing values (mean/median/mode hoặc drop)
    # TODO: mã hóa biến phân loại (One-Hot / Label Encoding)
    # TODO: chuẩn hóa biến số (StandardScaler) và lưu scaler
    pass


def build_dataset():
    # TODO: chạy load -> explore -> preprocess
    # TODO: lưu X_clinical.npy, y_clinical.npy
    pass


if __name__ == "__main__":
    build_dataset()
