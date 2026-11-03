"""THÀNH VIÊN 2 – PPG
Bước 5-6: xây dựng mô hình, đánh giá, lưu
Output: ppg_model.pkl (và ppg_scaler.pkl nếu có)
"""

# TODO 5: Xây dựng mô hình phân tích tín hiệu phù hợp với nhãn dữ liệu
def build_model():
    # TODO: chọn mô hình (Regression / Random Forest / CNN...) phù hợp nhãn
    pass

def train(model, X_train, y_train):
    # TODO: huấn luyện mô hình
    pass


# TODO 6: Đánh giá kết quả và viết báo cáo phần PPG
def evaluate(model, X_test, y_test):
    # TODO: chỉ số phù hợp (MAE/RMSE nếu hồi quy; Accuracy/F1 nếu phân loại)
    # TODO: viết báo cáo phần PPG
    pass

def save_model(model, path="mohinh/PPG/ppg_model.pkl"):
    # TODO: lưu mô hình
    pass


if __name__ == "__main__":
    # TODO: load X_ppg.npy, y_ppg.npy -> build -> train -> evaluate -> save
    pass
