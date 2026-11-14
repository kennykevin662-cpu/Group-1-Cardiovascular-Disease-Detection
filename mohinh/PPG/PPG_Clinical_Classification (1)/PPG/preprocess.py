"""THÀNH VIÊN – PPG | Clinical Database Classification
Bước 1–3: đọc dữ liệu Clinical, khám phá, làm sạch và tạo đặc trưng.
Nguồn dữ liệu: UCI Heart Disease – processed.cleveland.data.
Đầu ra: X_ppg.npy, y_ppg.npy, feature_names_ppg.npy.
Lưu ý: đây là phân loại dữ liệu lâm sàng dạng bảng, không phải tín hiệu PPG.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "Clinical" / "processed.cleveland.data"
FEATURE_NAMES = [
    "age", "sex", "chest_pain_type", "resting_blood_pressure",
    "cholesterol", "fasting_blood_sugar", "resting_ecg",
    "max_heart_rate", "exercise_induced_angina", "oldpeak",
    "st_slope", "major_vessels", "thalassemia"
]
COLUMN_NAMES = FEATURE_NAMES + ["diagnosis"]


def load_dataset(path=DATA_PATH):
    """Đọc Cleveland CSV; dấu ? được xem là giá trị thiếu."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy đặt processed.cleveland.data vào "
            "PPG/dataset/Clinical/."
        )
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?", skipinitialspace=True)
    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna().copy()
    df["diagnosis"] = (df["diagnosis"] > 0).astype(int)
    print(f"Đọc {before} dòng; giữ lại {len(df)} dòng đầy đủ ({before-len(df)} dòng thiếu dữ liệu bị loại).")
    return df


def explore_data(df, output_dir=None):
    """Thống kê nhãn và đặc trưng; lưu biểu đồ khám phá."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir) if output_dir else BASE_DIR / "explore"
    output.mkdir(parents=True, exist_ok=True)
    print("\nPhân bố nhãn (0 = không mắc bệnh, 1 = có bệnh):")
    print(df["diagnosis"].value_counts().sort_index().to_string())
    print("\nThống kê mô tả:")
    print(df[FEATURE_NAMES].describe().round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["diagnosis"].value_counts().sort_index().plot.bar(ax=axes[0], title="Phân bố nhãn", rot=0)
    df["age"].plot.hist(bins=20, ax=axes[1], title="Phân bố tuổi")
    axes[0].set_xlabel("Nhãn (0: không bệnh, 1: có bệnh)")
    axes[1].set_xlabel("Tuổi")
    fig.tight_layout()
    fig.savefig(output / "phan_bo_clinical.png", dpi=140)
    plt.close(fig)

    corr = df[FEATURE_NAMES + ["diagnosis"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=7)
    ax.set_title("Tương quan giữa các đặc trưng Clinical")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "clinical_correlation.png", dpi=140)
    plt.close(fig)
    print(f"Đã lưu biểu đồ khám phá tại: {output}")


def preprocess_data(df):
    """Tạo X/y; giữ nguyên đơn vị gốc và thứ tự 13 đặc trưng."""
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = df["diagnosis"].to_numpy(dtype=np.int64)
    if not np.isfinite(X).all():
        raise ValueError("X còn chứa NaN/Inf sau bước làm sạch.")
    return X, y


def build_dataset():
    df = load_dataset()
    explore_data(df)
    X, y = preprocess_data(df)
    np.save(BASE_DIR / "X_ppg.npy", X)
    np.save(BASE_DIR / "y_ppg.npy", y)
    np.save(BASE_DIR / "feature_names_ppg.npy", np.asarray(FEATURE_NAMES))
    print(f"\nĐã lưu X_ppg.npy: {X.shape}; y_ppg.npy: {y.shape}")
    print("Số mẫu theo nhãn:", dict(zip(*np.unique(y, return_counts=True))))
    return X, y


if __name__ == "__main__":
    build_dataset()
