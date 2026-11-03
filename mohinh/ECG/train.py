"""THÀNH VIÊN 1 – ECG
Bước 4-6: xây dựng, huấn luyện, đánh giá, lưu mô hình
Output: ecg_model.pkl (và ecg_scaler.pkl nếu có)
"""

# TODO 4: Xây dựng và huấn luyện mô hình Machine Learning / Deep Learning
def build_model():
    # TODO: chọn mô hình (Random Forest / 1D-CNN / ResNet1D ...)
    pass

def train(model, X_train, y_train, X_val, y_val):
    # TODO: huấn luyện, theo dõi loss/accuracy
    pass


# TODO 5: Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score
def evaluate(model, X_test, y_test):
    # TODO: tính 4 chỉ số trên + confusion matrix
    pass


# TODO 6: Lưu mô hình đã huấn luyện và viết báo cáo phần ECG
def save_model(model, path="mohinh/ECG/ecg_model.pkl"):
    # TODO: lưu mô hình (joblib / .h5 / .pt)
    # TODO: viết báo cáo phần ECG (mục tiêu, dữ liệu, phương pháp, kết quả)
    pass


if __name__ == "__main__":
    # TODO: load X_ecg.npy, y_ecg.npy -> build -> train -> evaluate -> save
    pass
