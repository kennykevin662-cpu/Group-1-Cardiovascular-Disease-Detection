"""THÀNH VIÊN 1 – ECG | Dataset: PTB-XL (PhysioNet)
Bước 1-3: tải dữ liệu, khám phá, tiền xử lý
Output: X_ecg.npy, y_ecg.npy (lưu trong mohinh/ECG/)
"""

import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # chỉ lưu ảnh, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import skew, kurtosis
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FS = 100  # dùng bản 100Hz (records100)
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# TODO 1: Tìm hiểu và tải dataset PTB-XL (đặt vào dataset/ECG/)
def load_dataset(path="dataset/ECG"):
    """
    Đọc ptbxl_database.csv và scp_statements.csv.
    Trả về (df, base_dir):
      - df: mỗi dòng là 1 bản ghi, có cột 'superclass' (list nhãn gốc:
            NORM/MI/STTC/CD/HYP) và cột 'y' (0 = bình thường, 1 = bất thường)
      - base_dir: thư mục chứa dữ liệu (để đọc file tín hiệu wfdb)
    """
    base = Path(path)
    if not base.is_absolute():
        base = ROOT / base
    csv = next(base.rglob("ptbxl_database.csv"), None)
    if csv is None:
        raise FileNotFoundError(
            f"Không tìm thấy ptbxl_database.csv trong {base}. "
            "Hãy tải PTB-XL và giải nén vào dataset/ECG/.")
    base = csv.parent

    df = pd.read_csv(csv, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(base / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]  # chỉ giữ mã chẩn đoán

    def to_superclass(codes):
        return sorted({scp.loc[c, "diagnostic_class"]
                       for c in codes if c in scp.index})

    df["superclass"] = df["scp_codes"].apply(to_superclass)
    df = df[df["superclass"].str.len() > 0].copy()
    df["y"] = df["superclass"].apply(lambda s: 0 if s == ["NORM"] else 1)
    print(f"Đã đọc {len(df)} bản ghi có nhãn chẩn đoán từ {base}")
    return df, base


# TODO 2: Khám phá dữ liệu, phân tích nhãn và đặc trưng
def explore_data(df, base_dir):
    """Thống kê nhãn, tuổi/giới tính và vẽ ECG mẫu. Ảnh lưu ở mohinh/ECG/explore/."""
    out = OUT_DIR / "explore"
    out.mkdir(exist_ok=True)

    # --- Thống kê nhãn ---
    dem = df["superclass"].explode().value_counts().reindex(SUPERCLASSES)
    print("\n=== Số bản ghi theo nhãn (1 bản ghi có thể có nhiều nhãn) ===")
    print(dem.to_string())
    print("\n=== Nhị phân: 0 = bình thường, 1 = bất thường ===")
    print(df["y"].value_counts().sort_index().to_string())

    # --- Tuổi / giới tính (PTB-XL ghi tuổi >89 thành 300 nên lọc ra) ---
    tuoi = df.loc[df["age"] < 120, "age"]
    print(f"\nTuổi: trung bình {tuoi.mean():.1f}, "
          f"nhỏ nhất {tuoi.min():.0f}, lớn nhất {tuoi.max():.0f}")
    gt = df["sex"].map({0: "Nam", 1: "Nữ"}).value_counts()
    print("Giới tính:", gt.to_dict())

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    dem.plot.bar(ax=ax[0], title="Số bản ghi theo nhãn")
    tuoi.plot.hist(bins=30, ax=ax[1], title="Phân bố tuổi")
    gt.plot.bar(ax=ax[2], title="Giới tính")
    fig.tight_layout()
    fig.savefig(out / "phan_bo.png", dpi=120)
    plt.close(fig)

    # --- Vẽ tín hiệu ECG mẫu theo từng nhãn (lead II, 10 giây) ---
    fig, axes = plt.subplots(len(SUPERCLASSES), 1,
                             figsize=(10, 2 * len(SUPERCLASSES)), sharex=True)
    t = np.arange(10 * FS) / FS
    for ax, nhan in zip(axes, SUPERCLASSES):
        mau = df[df["superclass"].apply(lambda s: s == [nhan])]
        if mau.empty:
            continue
        sig, _ = wfdb.rdsamp(str(base_dir / mau.iloc[0]["filename_lr"]))
        ax.plot(t, preprocess_signal(sig, FS)[:, 1], linewidth=0.8)
        ax.set_ylabel(nhan, rotation=0, labelpad=20)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Thời gian (giây) – lead II")
    fig.tight_layout()
    fig.savefig(out / "ecg_mau.png", dpi=120)
    plt.close(fig)
    print(f"Đã lưu ảnh khám phá vào {out}")


# TODO 3: Tiền xử lý tín hiệu ECG (lọc nhiễu, chuẩn hóa)
def preprocess_signal(signal, fs=FS, normalize=True):
    """
    signal: mảng (số_mẫu, số_lead).
    - Lọc bandpass 0.5–40Hz: bỏ baseline wander (<0.5Hz) và nhiễu cao tần
    - normalize=True: chuẩn hóa z-score theo từng lead
    """
    sos = butter(3, [0.5, 40.0], btype="band", fs=fs, output="sos")
    sig = sosfiltfilt(sos, signal, axis=0)
    if normalize:
        sig = (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-8)
    return sig


def extract_features(signal, fs=FS):
    """
    Chuyển tín hiệu (số_mẫu, 12 lead) thành vector đặc trưng cho Random Forest
    (predict.py cũng gọi hàm này nên train và dự đoán luôn đồng nhất).
    - 7 thống kê mỗi lead: mean, std, min, max, ptp, skew, kurtosis
    - 5 chỉ số nhịp tim ở lead II: HR, mean RR, std RR, RMSSD, số đỉnh R
    Lưu ý: dùng tín hiệu đã lọc nhưng CHƯA z-score, để giữ thông tin biên độ.
    """
    sig = preprocess_signal(signal, fs, normalize=False)
    feats = []
    for i in range(sig.shape[1]):
        x = sig[:, i]
        feats += [x.mean(), x.std(), x.min(), x.max(), np.ptp(x),
                  skew(x), kurtosis(x)]

    lead2 = sig[:, 1] if sig.shape[1] > 1 else sig[:, 0]
    peaks, _ = find_peaks(lead2, distance=int(0.3 * fs),
                          height=np.percentile(lead2, 90))
    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        feats += [60.0 / rr.mean(), rr.mean(), rr.std(),
                  np.sqrt(np.mean(np.diff(rr) ** 2)), len(peaks)]
    else:
        feats += [0, 0, 0, 0, len(peaks)]
    return np.nan_to_num(np.array(feats, dtype=np.float32))


def build_dataset():
    """
    load -> explore -> preprocess. Chia train/val/test theo strat_fold
    (khuyến nghị của PTB-XL): fold 1-8 train, fold 9 val, fold 10 test.
    Lưu X_ecg.npy, y_ecg.npy và folds_ecg.npy (để train.py chia lại).
    """
    df, base = load_dataset()
    explore_data(df, base)

    X, y, folds = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Trích đặc trưng"):
        try:
            sig, _ = wfdb.rdsamp(str(base / row["filename_lr"]))
            X.append(extract_features(sig))
            y.append(int(row["y"]))
            folds.append(int(row["strat_fold"]))
        except Exception as e:
            print("Bỏ qua bản ghi lỗi:", row["filename_lr"], e)

    X, y, folds = np.array(X), np.array(y), np.array(folds)
    np.save(OUT_DIR / "X_ecg.npy", X)
    np.save(OUT_DIR / "y_ecg.npy", y)
    np.save(OUT_DIR / "folds_ecg.npy", folds)

    print(f"\nX: {X.shape} | y: {y.shape}")
    print(f"Train (fold 1-8): {(folds <= 8).sum()} | "
          f"Val (fold 9): {(folds == 9).sum()} | "
          f"Test (fold 10): {(folds == 10).sum()}")
    print(f"Bình thường: {(y == 0).sum()} | Bất thường: {(y == 1).sum()}")


if __name__ == "__main__":
    build_dataset()