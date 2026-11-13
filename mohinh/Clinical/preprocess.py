from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # chỉ lưu ảnh, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent          # mohinh/Clinical
ROOT = HERE.parents[1]                          # gốc dự án (chứa dataset/ và mohinh/)
DATA_DIR = ROOT / "dataset" / "Clinical"
DATA_FILE = "processed.cleveland.data"

X_PATH = HERE / "X_clinical.npy"
Y_PATH = HERE / "y_clinical.npy"
SCALER_PATH = HERE / "clinical_scaler.pkl"      # lưu cả scaler + one-hot encoder
EDA_DIR = HERE / "eda"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# TODO 1: Tìm hiểu và tải dataset Heart Disease (đặt vào dataset/Clinical/)
def load_dataset(path=DATA_DIR):
    """Đọc processed.cleveland.data (không header, thiếu là '?').
    `path` có thể là thư mục chứa file hoặc đường dẫn thẳng tới file."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    if path.is_dir():
        path = path / DATA_FILE
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    # Nhãn gốc num: 0 = không bệnh, 1-4 = có bệnh -> nhị phân hóa
    df["target"] = (df["num"] > 0).astype(int)
    return df.drop(columns=["num"])


# TODO 2: Khám phá và phân tích dữ liệu
def explore_data(df, save_dir=EDA_DIR, show=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== df.info() ===")
    df.info()
    print("\n=== df.describe() ===")
    print(df.describe().T.round(2))
    print("\n=== Giá trị thiếu ===")
    miss = df.isna().sum()
    print(miss[miss > 0] if miss.any() else "Không có")
    print("\n=== Phân bố nhãn target (0 = không bệnh, 1 = có bệnh) ===")
    print(df["target"].value_counts().sort_index())
    print(df["target"].value_counts(normalize=True).sort_index().round(3))

    # Phân bố nhãn
    fig, ax = plt.subplots(figsize=(4, 3))
    df["target"].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4c9", "#e67"])
    ax.set_title("Phân bố nhãn target")
    ax.set_xticklabels(["0: Không bệnh", "1: Có bệnh"], rotation=0)
    fig.tight_layout()
    fig.savefig(save_dir / "target_distribution.png", dpi=120)

    # Heatmap tương quan
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax)
    ax.set_title("Heatmap tương quan")
    fig.tight_layout()
    fig.savefig(save_dir / "correlation_heatmap.png", dpi=120)

    # Histogram các đặc trưng số, tô màu theo target
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        for t, color, name in [(0, "#4c9", "Không bệnh"), (1, "#e67", "Có bệnh")]:
            ax.hist(df.loc[df["target"] == t, col].dropna(), bins=20, alpha=0.6, color=color, label=name)
        ax.set_title(col)
        ax.legend()
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(save_dir / "numeric_histograms.png", dpi=120)

    # Biến phân loại theo target
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        pd.crosstab(df[col], df["target"]).plot(kind="bar", ax=ax, color=["#4c9", "#e67"], legend=False)
        ax.set_title(col)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(save_dir / "categorical_vs_target.png", dpi=120)

    print(f"\nĐã lưu biểu đồ vào: {save_dir}")
    if show:
        matplotlib.use("TkAgg", force=True)
        plt.show()
    plt.close("all")


# TODO 3: Xử lý dữ liệu thiếu, mã hóa và chuẩn hóa dữ liệu
def preprocess(df):
    """Trả về (X, y, scaler). scaler = ColumnTransformer đã fit:
    - biến số: điền median -> StandardScaler
    - biến phân loại: điền mode -> One-Hot"""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    scaler = ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        sparse_threshold=0,  # luôn trả về mảng đặc
    )
    X = scaler.fit_transform(df[FEATURES]).astype(np.float32)
    y = df["target"].to_numpy()
    print(f"Sau xử lý: X {X.shape}, y {y.shape}")
    return X, y, scaler


def build_dataset(explore=True):
    df = load_dataset()
    if explore:
        explore_data(df)
    X, y, scaler = preprocess(df)
    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Đã lưu: {X_PATH.name}, {Y_PATH.name}, {SCALER_PATH.name} (trong {HERE})")
    return X, y


if __name__ == "__main__":
    build_dataset()