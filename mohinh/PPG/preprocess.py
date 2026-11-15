"""THÀNH VIÊN 2 – PPG | Dataset: PPG-BP (Guangdong Provincial People's Hospital / PhysioNet)
Đã được đồng bộ hóa & hoàn thiện bởi Thành viên 4 để tương thích với dữ liệu thực tế trong dataset/PPG/.

Quy trình:
1. Đọc dữ liệu PPG (.txt trong Data File/0_subject/) và nhãn nhịp tim (PPG-BP dataset.xlsx).
2. Tiền xử lý: lọc thông dải 0.5–8.0 Hz, chuẩn hóa Z-score.
3. Trích xuất đặc trưng (16 chiều): thống kê biên độ, đỉnh tâm thu, nhịp tim ước tính, HRV (RMSSD).
4. Lưu X_ppg.npy, y_ppg.npy, groups_ppg.npy trong mohinh/PPG/.
"""
import glob
import os
import sys
from pathlib import Path

# Đảm bảo console Windows không bị UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import kurtosis, skew
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = ROOT / "dataset" / "PPG"
DEFAULT_FS = 1000  # Bộ dữ liệu PPG-BP thu nhận ở tần số 1000 Hz


def load_dataset(path=DATA_DIR):
    """Đọc dữ liệu PPG-BP gồm các tệp .txt và file bảng PPG-BP dataset.xlsx.
    Nếu có các file BIDMC (*Signals.csv), hàm cũng sẽ tự động nhận diện."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path

    # 1. Kiểm tra bộ dữ liệu PPG-BP thực tế
    excel_candidates = list(path.rglob("PPG-BP dataset.xlsx"))
    txt_candidates = list(path.rglob("0_subject/*.txt")) + list(path.rglob("*.txt"))

    if excel_candidates and txt_candidates:
        excel_path = excel_candidates[0]
        # Bảng Excel: dòng thứ 2 (header=1) chứa tiêu đề cột
        df_meta = pd.read_excel(excel_path, header=1)
        # Chuẩn hóa cột subject_ID và Heart Rate
        df_meta = df_meta.dropna(subset=["subject_ID", "Heart Rate(b/m)"])
        meta_dict = {}
        for _, row in df_meta.iterrows():
            sub_id = int(row["subject_ID"])
            hr = float(row["Heart Rate(b/m)"])
            meta_dict[sub_id] = {
                "hr": hr,
                "sbp": float(row.get("Systolic Blood Pressure(mmHg)", np.nan)),
                "dbp": float(row.get("Diastolic Blood Pressure(mmHg)", np.nan)),
                "hypertension": str(row.get("Hypertension", "Unknown")),
            }

        records = []
        for txt_file in sorted(txt_candidates):
            fname = txt_file.stem
            # Tên file dạng: 100_1.txt -> subject_ID là 100
            try:
                sub_id = int(fname.split("_")[0])
            except ValueError:
                continue

            if sub_id not in meta_dict:
                continue

            try:
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().split()
                sig = np.array([float(x) for x in content if x], dtype=np.float32)
                if len(sig) < 100:
                    continue
                records.append({
                    "record_id": fname,
                    "subject_id": sub_id,
                    "ppg": sig,
                    "hr": meta_dict[sub_id]["hr"],
                    "fs": DEFAULT_FS,
                })
            except Exception as e:
                print(f"Bỏ qua file lỗi {txt_file.name}: {e}")

        print(f"Đã đọc {len(records)} bản ghi PPG từ bộ dữ liệu PPG-BP.")
        return records

    # 2. Hỗ trợ dự phòng nếu có file dạng BIDMC Signals.csv
    signal_files = sorted(path.rglob("*Signals.csv"))
    if signal_files:
        records = []
        for sfile in signal_files:
            nfile = sfile.parent / sfile.name.replace("Signals.csv", "Numerics.csv")
            if not nfile.exists():
                continue
            sdf = pd.read_csv(sfile)
            ndf = pd.read_csv(nfile)
            ppg_col = [c for c in sdf.columns if "pleth" in c.lower() or "ppg" in c.lower()]
            hr_col = [c for c in ndf.columns if "hr" in c.lower() or "pulse" in c.lower()]
            if ppg_col and hr_col:
                records.append({
                    "record_id": sfile.stem,
                    "subject_id": sfile.stem,
                    "ppg": pd.to_numeric(sdf[ppg_col[0]], errors="coerce").dropna().to_numpy(),
                    "hr": pd.to_numeric(ndf[hr_col[0]], errors="coerce").dropna().median(),
                    "fs": 125,
                })
        return records

    raise FileNotFoundError(f"Không tìm thấy dữ liệu PPG hợp lệ trong {path}")


def preprocess_signal(signal, fs=DEFAULT_FS):
    """Lọc thông dải 0.5–8.0 Hz và chuẩn hóa Z-score tín hiệu PPG."""
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size < 16:
        return x

    # Thiết kế bộ lọc Butterworth 0.5 - 8.0 Hz
    high_cut = min(8.0, fs / 2.0 - 0.5)
    low_cut = 0.5
    if low_cut >= high_cut:
        high_cut = low_cut + 1.0

    sos = butter(3, [low_cut, high_cut], btype="bandpass", fs=fs, output="sos")
    try:
        filtered = sosfiltfilt(sos, x)
    except Exception:
        from scipy.signal import sosfilt
        filtered = sosfilt(sos, x)

    std = np.std(filtered)
    if std < 1e-8:
        return np.zeros_like(filtered)
    return (filtered - np.mean(filtered)) / std


def extract_features(signal, fs=DEFAULT_FS):
    """Trích xuất 16 đặc trưng dạng bảng từ tín hiệu PPG:
    - 0-8: Các thống kê biên độ (mean, std, median, min, max, ptp, rms, skew, kurtosis)
    - 9: Số lượng đỉnh xung phát hiện
    - 10-12: Khoảng cách RR/IBI trung bình, độ lệch chuẩn RR, biên độ đỉnh trung bình
    - 13: Nhịp tim ước tính từ khoảng cách đỉnh (estimated_hr)
    - 14: Biến thiên nhịp tim RMSSD
    - 15: Khoảng biến thiên biên độ đỉnh (ptp peak amplitudes)
    """
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(16, dtype=np.float32)

    # Khoảng cách tối thiểu giữa 2 đỉnh xung: tương đương nhịp tim tối đa 220 bpm (60/220 s)
    min_dist = max(1, int((60.0 / 220.0) * fs))
    prominence = max(0.2 * np.std(x), 0.1)

    peaks, _ = find_peaks(x, distance=min_dist, prominence=prominence)
    intervals = np.diff(peaks) / float(fs)
    # Lọc khoảng RR hợp lệ (nhịp tim sinh học từ 35 bpm đến 200 bpm)
    valid_intervals = intervals[(intervals >= 60.0 / 220.0) & (intervals <= 60.0 / 35.0)]

    estimated_hr = float(60.0 / np.median(valid_intervals)) if valid_intervals.size else 0.0
    peak_amplitudes = x[peaks] if peaks.size else np.array([], dtype=np.float32)
    interval_rmssd = (
        float(np.sqrt(np.mean(np.diff(valid_intervals) ** 2)))
        if valid_intervals.size >= 2 else 0.0
    )

    features = [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.median(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.ptp(x)),
        float(np.sqrt(np.mean(x ** 2))),
        float(skew(x)) if x.size > 2 and np.std(x) > 0 else 0.0,
        float(kurtosis(x)) if x.size > 3 and np.std(x) > 0 else 0.0,
        int(peaks.size),
        float(np.mean(valid_intervals)) if valid_intervals.size else 0.0,
        float(np.std(valid_intervals)) if valid_intervals.size else 0.0,
        float(np.mean(peak_amplitudes)) if peak_amplitudes.size else 0.0,
        float(estimated_hr),  # INDEX 13: ESTIMATED HR TỪ ĐỈNH
        float(interval_rmssd),  # INDEX 14: RMSSD
        float(np.ptp(peak_amplitudes)) if peak_amplitudes.size else 0.0,
    ]
    return np.nan_to_num(np.array(features, dtype=np.float32))


def build_dataset():
    """Tải dữ liệu, tiền xử lý và lưu X_ppg.npy, y_ppg.npy."""
    records = load_dataset()
    X, y, groups = [], [], []

    for r in tqdm(records, desc="Trích đặc trưng PPG"):
        clean_sig = preprocess_signal(r["ppg"], fs=r["fs"])
        feats = extract_features(clean_sig, fs=r["fs"])
        X.append(feats)
        y.append(float(r["hr"]))
        groups.append(r["subject_id"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    groups = np.array(groups)

    np.save(HERE / "X_ppg.npy", X)
    np.save(HERE / "y_ppg.npy", y)
    np.save(HERE / "groups_ppg.npy", groups)

    print(f"\nĐã lưu: X_ppg {X.shape}, y_ppg {y.shape}, groups {groups.shape} vào {HERE}")
    return X, y, groups


if __name__ == "__main__":
    build_dataset()
