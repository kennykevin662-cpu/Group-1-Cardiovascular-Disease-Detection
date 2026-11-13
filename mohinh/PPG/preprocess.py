"""THÀNH VIÊN 2 – PPG | Dataset: BIDMC PPG and Respiration (PhysioNet)
Bước 1-4: tải dữ liệu, khám phá, tiền xử lý, trích đặc trưng
Output: X_ppg.npy, y_ppg.npy (lưu trong mohinh/PPG/)
"""

# TODO 1: Tìm hiểu và tải dataset BIDMC PPG and Respiration (đặt vào dataset/PPG/)
def load_dataset(path="dataset/PPG"):
    # Đọc các cặp file BIDMC dạng *_Signals.csv và *_Numerics.csv.
    import os
    import glob
    import pandas as pd

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {path}. "
            "Hãy đặt các file BIDMC vào thư mục này."
        )

    signal_files = sorted(glob.glob(os.path.join(path, "*Signals.csv")))
    if not signal_files:
        raise FileNotFoundError(
            f"Không tìm thấy file *Signals.csv trong {path}. "
            "Hàm này đang đọc dữ liệu BIDMC dạng CSV, chưa đọc trực tiếp WFDB."
        )

    records = []
    for signal_file in signal_files:
        base = signal_file[:-len("Signals.csv")]
        numerics_file = base + "Numerics.csv"
        if not os.path.isfile(numerics_file):
            print(f"Bỏ qua {signal_file}: thiếu file Numerics.csv tương ứng.")
            continue

        signals = pd.read_csv(signal_file)
        numerics = pd.read_csv(numerics_file)

        ppg_candidates = [
            col for col in signals.columns
            if any(key in str(col).lower() for key in ("pleth", "ppg"))
        ]
        if not ppg_candidates:
            print(f"Bỏ qua {signal_file}: không tìm thấy cột PLETH/PPG.")
            continue

        hr_candidates = [
            col for col in numerics.columns
            if any(key in str(col).lower() for key in ("pulse", "hr", "heart rate"))
        ]
        if not hr_candidates:
            print(f"Bỏ qua {numerics_file}: không tìm thấy cột HR/PULSE.")
            continue

        ppg_col = ppg_candidates[0]
        hr_col = hr_candidates[0]
        records.append({
            "record_id": os.path.basename(base.rstrip("_")),
            "signals": signals,
            "numerics": numerics,
            "ppg": pd.to_numeric(signals[ppg_col], errors="coerce").to_numpy(),
            "hr": pd.to_numeric(numerics[hr_col], errors="coerce").to_numpy(),
            "ppg_column": ppg_col,
            "hr_column": hr_col,
        })

    if not records:
        raise ValueError("Không đọc được bản ghi BIDMC hợp lệ từ các file CSV.")
    return records


# TODO 2: Khám phá và trực quan hóa tín hiệu PPG
def explore_data(data):
    # Vẽ tín hiệu PPG, phổ tần số và thống kê dữ liệu.
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import periodogram

    os.makedirs("outputs", exist_ok=True)
    print(f"Số bản ghi: {len(data)}")

    for index, record in enumerate(data):
        signal = np.asarray(record["ppg"], dtype=float)
        signal = signal[np.isfinite(signal)]
        if signal.size == 0:
            continue

        fs = 125.0
        time = np.arange(signal.size) / fs
        frequencies, power = periodogram(signal, fs=fs)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        axes[0].plot(time, signal, linewidth=0.8)
        axes[0].set_title(f"PPG waveform - {record['record_id']}")
        axes[0].set_xlabel("Thời gian (giây)")
        axes[0].set_ylabel("Biên độ")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(frequencies, power, linewidth=0.8)
        axes[1].set_xlim(0, 10)
        axes[1].set_title("Phổ tần số PPG")
        axes[1].set_xlabel("Tần số (Hz)")
        axes[1].set_ylabel("Công suất")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(f"outputs/ppg_explore_{index + 1}.png", dpi=150)
        plt.close(fig)

        hr = np.asarray(record["hr"], dtype=float)
        hr = hr[np.isfinite(hr)]
        print(
            f"{record['record_id']}: {signal.size} mẫu PPG; "
            f"HR hợp lệ: {hr.size}; "
            f"HR trung bình: {np.mean(hr):.2f} BPM" if hr.size else
            f"{record['record_id']}: {signal.size} mẫu PPG; không có HR hợp lệ."
        )


# TODO 3: Tiền xử lý tín hiệu, lọc nhiễu
def preprocess_signal(signal, fs=125):
    # Lọc thông dải 0.5–8 Hz và chuẩn hóa Z-score.
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(signal, dtype=float).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size < 16:
        raise ValueError("Tín hiệu PPG quá ngắn để tiền xử lý.")
    if fs <= 16:
        raise ValueError("Tần số lấy mẫu phải lớn hơn 16 Hz để lọc đến 8 Hz.")

    sos = butter(3, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    try:
        filtered = sosfiltfilt(sos, x)
    except ValueError:
        # Với đoạn tín hiệu ngắn, dùng lọc tiến để tránh lỗi độ dài padding.
        from scipy.signal import sosfilt
        filtered = sosfilt(sos, x)

    mean = np.mean(filtered)
    std = np.std(filtered)
    if std < 1e-8:
        return np.zeros_like(filtered)
    return (filtered - mean) / std


# TODO 4: Trích xuất đặc trưng như nhịp mạch, khoảng cách giữa các đỉnh
def extract_features(signal, fs=125):
    # Trích xuất đặc trưng thống kê, đỉnh mạch, nhịp tim và HRV.
    import numpy as np
    from scipy.signal import find_peaks
    from scipy.stats import skew, kurtosis

    x = np.asarray(signal, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Tín hiệu rỗng, không thể trích xuất đặc trưng.")

    peaks, _ = find_peaks(
        x,
        distance=max(1, int(0.30 * fs)),
        prominence=max(0.1 * np.std(x), 1e-6)
    )
    intervals = np.diff(peaks) / float(fs)
    valid_intervals = intervals[
        (intervals >= 60.0 / 220.0) & (intervals <= 60.0 / 30.0)
    ]

    estimated_hr = (
        60.0 / np.median(valid_intervals)
        if valid_intervals.size else 0.0
    )
    peak_amplitudes = x[peaks] if peaks.size else np.array([], dtype=float)
    interval_rmssd = (
        np.sqrt(np.mean(np.diff(valid_intervals) ** 2))
        if valid_intervals.size >= 2 else 0.0
    )

    features = [
        np.mean(x), np.std(x), np.median(x), np.min(x), np.max(x),
        np.ptp(x), np.sqrt(np.mean(x ** 2)),
        skew(x) if x.size > 2 and np.std(x) > 0 else 0.0,
        kurtosis(x) if x.size > 3 and np.std(x) > 0 else 0.0,
        peaks.size,
        np.mean(valid_intervals) if valid_intervals.size else 0.0,
        np.std(valid_intervals) if valid_intervals.size else 0.0,
        np.mean(peak_amplitudes) if peak_amplitudes.size else 0.0,
        estimated_hr,
        interval_rmssd,
        np.ptp(peak_amplitudes) if peak_amplitudes.size else 0.0,
    ]
    return np.nan_to_num(np.asarray(features, dtype=float))


def build_dataset():
    # Đọc dữ liệu, khám phá, chia thành cửa sổ và lưu X, y.
    import os
    import numpy as np

    records = load_dataset()
    explore_data(records)

    fs = 125
    window_seconds = 8
    overlap = 0.5
    window_size = int(window_seconds * fs)
    step = int(window_size * (1 - overlap))

    X, y, groups = [], [], []
    for record in records:
        ppg = np.asarray(record["ppg"], dtype=float)
        hr = np.asarray(record["hr"], dtype=float)

        # BIDMC Numerics thường lấy mẫu chậm hơn PPG; ánh xạ HR lên trục thời gian.
        if hr.size == 0 or ppg.size < window_size:
            continue
        ppg_duration = ppg.size / fs
        hr_time = np.linspace(0, ppg_duration, num=hr.size, endpoint=False)
        ppg_time = np.arange(ppg.size) / fs

        for start in range(0, ppg.size - window_size + 1, step):
            end = start + window_size
            segment = ppg[start:end]
            start_time = start / fs
            end_time = end / fs
            mask = (hr_time >= start_time) & (hr_time < end_time)
            hr_values = hr[mask]
            hr_values = hr_values[
                np.isfinite(hr_values) & (hr_values >= 30) & (hr_values <= 220)
            ]
            if hr_values.size == 0 or not np.all(np.isfinite(segment)):
                continue

            clean = preprocess_signal(segment, fs=fs)
            X.append(extract_features(clean, fs=fs))
            y.append(float(np.median(hr_values)))
            groups.append(record["record_id"])

    if not X:
        raise ValueError(
            "Không tạo được mẫu dữ liệu. Hãy kiểm tra định dạng CSV, "
            "cột PPG/HR và độ dài bản ghi."
        )

    output_dir = "mohinh/PPG"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_ppg.npy"), np.vstack(X))
    np.save(os.path.join(output_dir, "y_ppg.npy"), np.asarray(y, dtype=float))
    np.save(os.path.join(output_dir, "groups_ppg.npy"), np.asarray(groups))
    print(f"Đã lưu X_ppg.npy: {len(X)} mẫu, {len(X[0])} đặc trưng.")
    print(f"Đã lưu y_ppg.npy: {len(y)} nhãn BPM.")


if __name__ == "__main__":
    build_dataset()
