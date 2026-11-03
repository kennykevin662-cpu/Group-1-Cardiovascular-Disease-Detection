"""THÀNH VIÊN 3 – Clinical
Bước 4-6: xây dựng, huấn luyện, đánh giá, lưu mô hình
Output: clinical_model.pkl
"""

# TODO 4: Xây dựng mô hình phân loại nguy cơ/bệnh tim theo nhãn dataset
def build_model():
    # TODO: thử Logistic Regression, Random Forest, XGBoost...
    pass

def train(model, X_train, y_train):
    # TODO: huấn luyện, cross-validation, tinh chỉnh tham số
    pass


# TODO 5: Đánh giá mô hình bằng Accuracy, Precision, Recall, F1-score
def evaluate(model, X_test, y_test):
    # TODO: tính 4 chỉ số trên + confusion matrix
    pass


# TODO 6: Lưu mô hình và viết báo cáo phần Clinical
def save_model(model, path="mohinh/Clinical/clinical_model.pkl"):
    # TODO: lưu mô hình
    # TODO: viết báo cáo phần Clinical
    pass


if __name__ == "__main__":
    # TODO: load X_clinical.npy, y_clinical.npy -> build -> train -> evaluate -> save
    pass
