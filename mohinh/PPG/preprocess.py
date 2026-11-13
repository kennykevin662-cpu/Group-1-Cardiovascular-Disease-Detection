""THÀNH VIÊN 2 – PPG | Dataset: BIDMC PPG and Respiration (PhysioNet)
Bước 1-4: tải dữ liệu, khám phá, tiền xử lý, trích đặc trưng
Output: X_ppg.npy, y_ppg.npy (lưu trong mohinh/PPG/)
 
Tải dataset: https://physionet.org/content/bidmc/1.0.0/
Giải nén vào dataset/PPG/ (hỗ trợ cả file WFDB: bidmc01.dat/.hea, bidmc01n.dat/.hea
và file CSV: bidmc_01_Signals.csv, bidmc_01_Numerics.csv).
 
Bài toán: HỒI QUY – dự đoán nhịp tim (HR, bpm) cho mỗi cửa sổ tín hiệu PPG.
"""
import glob
import os
 
import numpy as np
from scipy import signal as sp
from scipy.stats import kurtosis, skew
 
FS = 125                 # tần số lấy mẫu của BIDMC (Hz)
WINDOW_SEC = 10          # độ dài cửa sổ (giây)
STEP_SEC = 5             # bước trượt (giây)
OUT_DIR = "mohinh/PPG"
 
FEATURE_NAMES = [
    "hr_peaks",        # nhịp tim tính từ khoảng cách đỉnh (bpm)
    "ibi_mean",        # khoảng peak-to-peak trung bình (s)
    "ibi_median",
    "ibi_std",         # ~ SDNN
    "rmssd",           # HRV: căn bậc hai trung bình bình phương hiệu liên tiếp (s)
    "pnn50",           # tỉ lệ khoảng liên tiếp lệch > 50 ms
    "n_peaks",
    "amp_mean",        # biên độ đỉnh
    "amp_std",
    "sig_skew",
    "sig_kurt",
    "dom_freq_bpm",    # tần số trội trong dải 0.5-3.5 Hz đổi ra bpm
    "spec_entropy",    # entropy phổ
    "pulse_width",     # độ rộng trung bình giữa 2 đáy kề nhau (s)
]
 
 
# TODO 1: Tìm hiểu và tải dataset BIDMC PPG and Respiration (đặt vào dataset/PPG/)
def _norm(names):
    return [n.strip().strip(",").strip().upper() for n in names]
 
 
def load_dataset(path="dataset/PPG"):
    """Trả về list dict: {id, ppg, fs, hr, resp, spo2, fs_num}.
    hr/resp/spo2 là số liệu 1 Hz (numerics)."""
    records = []
 
    # --- Cách 1: WFDB ---
    hea_files = sorted(glob.glob(os.path.join(path, "**", "bidmc[0-9][0-9].hea"), recursive=True))
    if hea_files:
        import wfdb
        for h in hea_files:
            base = h[:-4]
            rec = wfdb.rdrecord(base)
            names = _norm(rec.sig_name)
            ppg = rec.p_signal[:, names.index("PLETH")]
            item = {"id": os.path.basename(base), "ppg": ppg, "fs": int(rec.fs),
                    "hr": None, "resp": None, "spo2": None, "fs_num": 1}
            if os.path.exists(base + "n.hea"):
                num = wfdb.rdrecord(base + "n")
                nn = _norm(num.sig_name)
                item["fs_num"] = num.fs
                for key, col in (("hr", "HR"), ("resp", "RESP"), ("spo2", "SPO2")):
                    if col in nn:
                        item[key] = num.p_signal[:, nn.index(col)]
            records.append(item)
        return records
 
    # --- Cách 2: CSV ---
    import pandas as pd
    sig_files = sorted(glob.glob(os.path.join(path, "**", "bidmc_[0-9][0-9]_Signals.csv"), recursive=True))
    for f in sig_files:
        df = pd.read_csv(f)
        df.columns = [c.strip().upper() for c in df.columns]
        t = df["TIME [S]"].values
        fs = int(round(1.0 / np.median(np.diff(t))))
        item = {"id": os.path.basename(f).replace("_Signals.csv", ""),
                "ppg": df["PLETH"].values.astype(float), "fs": fs,
                "hr": None, "resp": None, "spo2": None, "fs_num": 1}
        nf = f.replace("_Signals.csv", "_Numerics.csv")
        if os.path.exists(nf):
            dn = pd.read_csv(nf)
            dn.columns = [c.strip().upper() for c in dn.columns]
            for key, col in (("hr", "HR"), ("resp", "RESP"), ("spo2", "SPO2")):
                if col in dn.columns:
                    item[key] = dn[col].values.astype(float)
        records.append(item)
 
    if not records:
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu BIDMC trong '{path}'. Hãy tải dataset từ PhysioNet vào thư mục này.")
    return records
 
 
# TODO 2: Khám phá và trực quan hóa tín hiệu PPG
def explore_data(data, out_dir=OUT_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
 
    # Thống kê chung
    lengths = [len(r["ppg"]) / r["fs"] for r in data]
    fs_set = sorted({r["fs"] for r in data})
    print(f"Số bản ghi: {len(data)}")
    print(f"Tần số lấy mẫu: {fs_set} Hz")
    print(f"Độ dài (s): min={min(lengths):.0f}, max={max(lengths):.0f}, TB={np.mean(lengths):.0f}")
    all_hr = np.concatenate([r["hr"][~np.isnan(r["hr"])] for r in data if r["hr"] is not None])
    print(f"Nhãn HR (bpm): min={all_hr.min():.0f}, max={all_hr.max():.0f}, "
          f"TB={all_hr.mean():.1f}, std={all_hr.std():.1f}")
 
    r = data[0]
    fs = r["fs"]
    seg = r["ppg"][: 10 * fs]
    t = np.arange(len(seg)) / fs
 
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(t, seg)
    ax[0].set_title(f"PPG thô – {r['id']} (10 s đầu)")
    ax[0].set_xlabel("Thời gian (s)")
 
    f, pxx = sp.welch(r["ppg"] - np.mean(r["ppg"]), fs, nperseg=8 * fs)
    ax[1].semilogy(f, pxx)
    ax[1].set_xlim(0, 10)
    ax[1].set_title("Phổ công suất (Welch)")
    ax[1].set_xlabel("Tần số (Hz)")
 
    ax[2].hist(all_hr, bins=40)
    ax[2].set_title("Phân bố nhãn HR")
    ax[2].set_xlabel("bpm")
 
    plt.tight_layout()
    path = os.path.join(out_dir, "figures", "explore_ppg.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print("Đã lưu hình:", path)
 
 
# TODO 3: Tiền xử lý tín hiệu, lọc nhiễu
def preprocess_signal(signal, fs=FS):
    """Lọc bandpass 0.5-8 Hz, hạn chế nhiễu chuyển động (cắt biên độ bất thường), chuẩn hóa z-score."""
    x = np.asarray(signal, dtype=float)
    x = np.nan_to_num(x, nan=np.nanmedian(x) if np.any(~np.isnan(x)) else 0.0)
 
    sos = sp.butter(4, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    x = sp.sosfiltfilt(sos, x)
 
    # Nhiễu chuyển động: cắt các đoạn có biên độ lớn bất thường (theo MAD)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-8
    x = np.clip(x, med - 6 * mad, med + 6 * mad)
 
    return (x - np.mean(x)) / (np.std(x) + 1e-8)
 
 
# TODO 4: Trích xuất đặc trưng như nhịp mạch, khoảng cách giữa các đỉnh
def extract_features(signal, fs=FS):
    """Trả về vector đặc trưng (theo FEATURE_NAMES) hoặc None nếu không đủ đỉnh."""
    x = np.asarray(signal, dtype=float)
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
 
    peaks, _ = sp.find_peaks(x, distance=int(0.3 * fs), prominence=0.3)
    if len(peaks) < 3:
        return None
 
    ibi = np.diff(peaks) / fs
    hr_peaks = 60.0 / np.mean(ibi)
    diff_ibi = np.diff(ibi)
    rmssd = np.sqrt(np.mean(diff_ibi ** 2)) if len(diff_ibi) else 0.0
    pnn50 = np.mean(np.abs(diff_ibi) > 0.05) if len(diff_ibi) else 0.0
 
    amps = x[peaks]
 
    # độ rộng xung: khoảng cách giữa 2 đáy kề nhau
    valleys, _ = sp.find_peaks(-x, distance=int(0.3 * fs), prominence=0.3)
    pulse_width = np.mean(np.diff(valleys)) / fs if len(valleys) >= 2 else np.mean(ibi)
 
    # phổ
    f, pxx = sp.welch(x, fs, nperseg=min(len(x), 8 * fs))
    band = (f >= 0.5) & (f <= 3.5)
    fb, pb = f[band], pxx[band]
    dom_bpm = fb[np.argmax(pb)] * 60.0
    p_norm = pb / (pb.sum() + 1e-12)
    spec_entropy = -np.sum(p_norm * np.log(p_norm + 1e-12)) / np.log(len(p_norm))
 
    return np.array([
        hr_peaks, np.mean(ibi), np.median(ibi), np.std(ibi), rmssd, pnn50, len(peaks),
        np.mean(amps), np.std(amps), skew(x), kurtosis(x), dom_bpm, spec_entropy, pulse_width,
    ], dtype=float)
 
 
def build_dataset(path="dataset/PPG", out_dir=OUT_DIR):
    data = load_dataset(path)
    explore_data(data, out_dir)
 
    X, y, groups = [], [], []
    win, step = WINDOW_SEC, STEP_SEC
    for gi, r in enumerate(data):
        if r["hr"] is None:
            continue
        fs = r["fs"]
        sig = preprocess_signal(r["ppg"], fs)
        n_sec = len(sig) // fs
        for start in range(0, n_sec - win + 1, step):
            seg = sig[start * fs:(start + win) * fs]
            hr_win = r["hr"][start:start + win]
            label = np.nanmean(hr_win) if np.any(~np.isnan(hr_win)) else np.nan
            if np.isnan(label) or label <= 0:
                continue
            feats = extract_features(seg, fs)
            if feats is None:
                continue
            X.append(feats)
            y.append(label)
            groups.append(gi)
 
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_ppg.npy"), X)
    np.save(os.path.join(out_dir, "y_ppg.npy"), y)
    np.save(os.path.join(out_dir, "groups_ppg.npy"), groups)  # mã bản ghi, để chia train/test theo bệnh nhân
    print(f"Đã lưu X_ppg.npy {X.shape}, y_ppg.npy {y.shape} vào {out_dir}/")
 
 
if __name__ == "__main__":
    build_dataset()
