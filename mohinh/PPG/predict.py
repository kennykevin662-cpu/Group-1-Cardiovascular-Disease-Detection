"""
THÀNH VIÊN 2 – PPG
Hàm dự đoán để Thành viên 4 gọi khi tích hợp vào giao diện.

predict(signal, fs=125) trả về nhịp mạch dự đoán (BPM).
Đầu vào phải là tín hiệu PPG thô, chưa lọc/chuẩn hóa.
"""
from pathlib import Path
import joblib
import numpy as np

try:
    from .preprocess import (
        preprocess_signal, extract_features, FEATURE_NAMES,
        DEFAULT_FS, WINDOW_SECONDS
    )
except ImportError:
    from preprocess import (
        preprocess_signal, extract_features, FEATURE_NAMES,
        DEFAULT_FS, WINDOW_SECONDS
    )

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ppg_model.pkl"
_MODEL_BUNDLE = None


def _load_bundle(model_path=MODEL_PATH):
    global _MODEL_BUNDLE
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy chạy preprocess.py và train.py trước."
        )
    if path.resolve() == MODEL_PATH.resolve() and _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    bundle = joblib.load(path)
    # Tương thích với file cũ chỉ lưu trực tiếp estimator.
    if isinstance(bundle, dict) and "model" in bundle:
        model_bundle = bundle
    else:
        model_bundle = {
            "model": bundle,
            "feature_names": FEATURE_NAMES,
            "fs": DEFAULT_FS,
            "window_seconds": WINDOW_SECONDS,
            "target": "PULSE (BPM)"
        }
    if path.resolve() == MODEL_PATH.resolve():
        _MODEL_BUNDLE = model_bundle
    return model_bundle


def predict(signal, fs=DEFAULT_FS, model_path=MODEL_PATH):
    """
    Dự đoán nhịp mạch từ một tín hiệu PPG.

    Parameters
    ----------
    signal : array-like
        Mảng tín hiệu PPG thô.
    fs : int
        Tần số lấy mẫu (BIDMC mặc định 125 Hz).
    model_path : str/Path
        Đường dẫn file ppg_model.pkl.

    Returns
    -------
    float
        Nhịp mạch dự đoán, đơn vị BPM.
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("Tín hiệu PPG đầu vào đang rỗng.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Tín hiệu PPG chứa NaN hoặc Inf.")

    bundle = _load_bundle(model_path)
    model = bundle["model"]
    expected_features = bundle.get("feature_names", FEATURE_NAMES)
    expected_fs = int(bundle.get("fs", DEFAULT_FS))
    window_seconds = int(bundle.get("window_seconds", WINDOW_SECONDS))

    if fs != expected_fs:
        raise ValueError(
            f"Mô hình được huấn luyện với fs={expected_fs} Hz, "
            f"nhưng đầu vào fs={fs} Hz. Hãy resample tín hiệu trước."
        )

    window_size = int(window_seconds * fs)
    if len(x) < window_size:
        raise ValueError(
            f"Cần ít nhất {window_seconds} giây PPG "
            f"({window_size} mẫu ở {fs} Hz)."
        )

    # Dùng cùng kích thước cửa sổ khi xây dựng dữ liệu huấn luyện.
    step_size = window_size // 2
    feature_rows = []
    for start in range(0, len(x) - window_size + 1, step_size):
        segment = x[start:start + window_size]
        try:
            clean = preprocess_signal(segment, fs)
            row = extract_features(clean, fs)
            feature_rows.append(row)
        except ValueError:
            continue

    if not feature_rows:
        raise ValueError("Không trích xuất được đặc trưng hợp lệ từ tín hiệu.")

    X = np.asarray(feature_rows, dtype=float)
    if list(expected_features) != FEATURE_NAMES:
        raise ValueError(
            "Danh sách/thứ tự đặc trưng trong model không khớp preprocess.py. "
            "Hãy huấn luyện lại model bằng train.py hiện tại."
        )
    if hasattr(model, "n_features_in_") and model.n_features_in_ != X.shape[1]:
        raise ValueError(
            f"Mô hình yêu cầu {model.n_features_in_} đặc trưng, "
            f"nhưng đầu vào có {X.shape[1]}."
        )

    predictions = np.asarray(model.predict(X), dtype=float).reshape(-1)
    predictions = predictions[np.isfinite(predictions)]
    if predictions.size == 0:
        raise ValueError("Mô hình không trả về dự đoán hợp lệ.")

    # Trung vị giúp giảm ảnh hưởng của một cửa sổ bất thường.
    bpm = float(np.median(predictions))
    if not 30 <= bpm <= 220:
        raise ValueError(
            f"Giá trị dự đoán {bpm:.2f} BPM nằm ngoài ngưỡng kiểm tra 30–220."
        )
    return round(bpm, 2)


if __name__ == "__main__":
    # Ví dụ kiểm tra giao diện hàm; cần có ppg_model.pkl đã huấn luyện.
    fs = DEFAULT_FS
    time = np.arange(fs * 16) / fs
    demo_signal = np.sin(2 * np.pi * 1.2 * time)

    try:
        print("Pulse rate:", predict(demo_signal, fs=fs), "BPM")
    except (FileNotFoundError, ValueError) as error:
        print("Lỗi:", error)
