"""
THÀNH VIÊN 4 – SYSTEM
Screening – Logic sàng lọc bệnh tim mạch.

Xử lý dữ liệu đầu vào và gọi mô hình tương ứng:
ECG / PPG / Clinical.
"""

import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:
    wfdb = None

from .hamhotro import (
    format_prediction,
    handle_error,
    validate_file,
    validate_input,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Hằng số chuẩn theo mohinh/ECG/preprocess.py (PTB-XL 100Hz, 12 leads)
ECG_FS = 100
ECG_EXPECTED_LEADS = 12
ECG_MIN_SAMPLES = 30  # Lớn hơn padlen=21 của bộ lọc Butterworth bậc 3 (sosfiltfilt)

# Danh sách 13 đặc trưng chuẩn của bộ dữ liệu UCI Heart Disease
CLINICAL_FEATURES: List[str] = [
    "age",       # Tuổi
    "sex",       # Giới tính (1 = nam, 0 = nữ)
    "cp",        # Loại đau ngực (0-3)
    "trestbps",  # Huyết áp lúc nghỉ (mm Hg)
    "chol",      # Cholesterol huyết thanh (mg/dl)
    "fbs",       # Đường huyết lúc đói > 120 mg/dl (1 = đúng, 0 = sai)
    "restecg",   # Kết quả điện tâm đồ lúc nghỉ (0-2)
    "thalach",   # Nhịp tim tối đa đạt được
    "exang",     # Đau thắt ngực khi gắng sức (1 = có, 0 = không)
    "oldpeak",   # Chênh xuống ST do gắng sức
    "slope",     # Độ dốc đoạn ST đỉnh gắng sức (0-2)
    "ca",        # Số mạch máu chính được nhuộm màu (0-3)
    "thal",      # Thalassemia (0-3)
]


def _load_predict_function(modality: str, candidate_funcs: Tuple[str, ...]) -> Callable[..., Any]:
    """
    Nạp động file mohinh/<modality>/predict.py với tên module riêng biệt
    để tránh xung đột giữa 3 file cùng tên predict.py.
    """
    predict_path = PROJECT_ROOT / "mohinh" / modality / "predict.py"
    if not predict_path.exists():
        raise FileNotFoundError(f"Chưa tìm thấy file {predict_path}")

    module_name = f"mohinh_{modality.lower()}_predict"
    spec = importlib.util.spec_from_file_location(module_name, str(predict_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể khởi tạo module từ {predict_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    for func_name in candidate_funcs:
        fn = getattr(module, func_name, None)
        if callable(fn):
            return fn

    raise AttributeError(
        f"File {predict_path} không có hàm dự đoán hợp lệ ({', '.join(candidate_funcs)})."
    )


def _read_wfdb_files(source: Any) -> np.ndarray:
    """
    Đọc tín hiệu PTB-XL định dạng WFDB (.hea + .dat) từ đường dẫn hoặc danh sách file upload.
    """
    if wfdb is None:
        raise ImportError("Thiếu thư viện 'wfdb' để đọc file PTB-XL (.hea/.dat). Hãy chạy: pip install wfdb")

    if isinstance(source, (str, Path)):
        record_base = str(Path(source).with_suffix(""))
        sig, _ = wfdb.rdsamp(record_base)
        return np.asarray(sig, dtype=np.float32)

    if isinstance(source, (list, tuple)):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            hea_stem = None
            for f in source:
                fname = Path(getattr(f, "name", "")).name
                if hasattr(f, "seek"):
                    f.seek(0)
                (tmp_path / fname).write_bytes(f.read())
                if fname.lower().endswith(".hea"):
                    hea_stem = tmp_path / Path(fname).stem
            if hea_stem is None:
                raise ValueError("Định dạng WFDB yêu cầu tải lên đồng thời cả 2 file: .hea và .dat.")
            sig, _ = wfdb.rdsamp(str(hea_stem))
            return np.asarray(sig, dtype=np.float32)

    raise ValueError("Để đọc file WFDB (.dat/.hea), vui lòng cung cấp đường dẫn file hoặc tải lên cả cặp .hea và .dat.")


def _load_signal_from_source(source: Any) -> np.ndarray:
    """
    Đọc tín hiệu từ ndarray, list, DataFrame, file CSV/NPY/TXT hoặc cặp file WFDB (.hea/.dat).
    """
    if isinstance(source, np.ndarray):
        return source.astype(np.float32)

    if isinstance(source, (list, tuple)):
        if len(source) > 0 and hasattr(source[0], "read"):
            return _read_wfdb_files(source)
        return np.asarray(source, dtype=np.float32)

    if isinstance(source, (pd.DataFrame, pd.Series)):
        return source.to_numpy(dtype=np.float32)

    is_valid, msg = validate_file(source, allowed_extensions=(".csv", ".npy", ".txt", ".hea", ".dat"))
    if not is_valid:
        raise ValueError(msg)

    filename = str(source) if isinstance(source, (str, Path)) else getattr(source, "name", "")
    ext = Path(filename).suffix.lower()

    if ext in {".hea", ".dat"}:
        return _read_wfdb_files(source)

    if hasattr(source, "seek"):
        source.seek(0)

    if ext == ".npy":
        arr = np.load(source, allow_pickle=False)
        return np.asarray(arr, dtype=np.float32)

    df = pd.read_csv(source, header=None)
    first_row_is_header = df.iloc[0].apply(
        lambda x: isinstance(x, str) and not x.replace(".", "", 1).replace("-", "", 1).isdigit()
    ).any()
    if first_row_is_header:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("File không chứa dữ liệu tín hiệu dạng số.")
    return numeric_df.to_numpy(dtype=np.float32)


def _normalize_ecg_shape(ecg_array: np.ndarray) -> np.ndarray:
    """
    Đảm bảo mảng ECG có dạng 2D (số_mẫu, số_lead) khớp với preprocess.py của Thành viên 1.
    """
    if ecg_array.ndim == 1:
        return ecg_array.reshape(-1, 1)
    if ecg_array.ndim == 2:
        if ecg_array.shape[0] == ECG_EXPECTED_LEADS and ecg_array.shape[1] > ECG_EXPECTED_LEADS:
            return ecg_array.T
        return ecg_array
    raise ValueError(f"Tín hiệu ECG phải là mảng 1D hoặc 2D, nhận được shape {ecg_array.shape}.")


def _estimate_heart_rate_1d(signal_1d: np.ndarray, fs: int) -> Dict[str, float]:
    """
    Ước tính nhanh nhịp tim (BPM) và chỉ số thống kê từ tín hiệu 1 kênh (Lead II ECG hoặc PPG).
    """
    sig = signal_1d.flatten()
    sig = sig[~np.isnan(sig)]
    if sig.size < fs:
        return {"signal_mean": float(np.mean(sig)) if sig.size else 0.0}

    norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    min_distance = max(1, int(0.35 * fs))
    peaks: List[int] = []

    for i in range(1, len(norm) - 1):
        if norm[i] > 0.8 and norm[i] > norm[i - 1] and norm[i] >= norm[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)

    features: Dict[str, float] = {
        "signal_mean": round(float(np.mean(sig)), 4),
        "signal_std": round(float(np.std(sig)), 4),
    }

    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / float(fs)
        mean_rr = float(np.mean(rr_intervals))
        if mean_rr > 0:
            hr = 60.0 / mean_rr
            if 35.0 <= hr <= 220.0:
                features["heart_rate"] = round(hr, 1)

    return features


# ============================================================
# TODO 1: SCREENING ECG
# ============================================================

def screen_ecg(signal: Any, fs: int = ECG_FS) -> Dict[str, Any]:
    """
    Thực hiện quy trình sàng lọc bệnh tim mạch từ tín hiệu ECG (chuẩn PTB-XL 100Hz, 12 leads).
    """
    try:
        raw_array = _load_signal_from_source(signal)
        ecg_array = _normalize_ecg_shape(raw_array)

        is_valid, msg = validate_input(ecg_array, expected_type=np.ndarray, min_length=ECG_MIN_SAMPLES)
        if not is_valid:
            return handle_error(ValueError(msg), context="ECG")

        if np.isnan(ecg_array).any():
            ecg_array = np.nan_to_num(ecg_array, nan=0.0)

        lead_ii = ecg_array[:, 1] if ecg_array.shape[1] >= 2 else ecg_array[:, 0]
        ecg_stats = _estimate_heart_rate_1d(lead_ii, fs=fs)

        predict_fn = _load_predict_function("ECG", ("predict_ecg", "predict", "run_prediction"))
        raw_result = predict_fn(ecg_array)

        formatted = format_prediction(raw_result, modality="ECG")
        formatted.setdefault("details", {})
        for k, v in ecg_stats.items():
            formatted["details"].setdefault(k, v)
        formatted["details"]["signal_shape"] = list(ecg_array.shape)
        formatted["details"]["num_leads"] = int(ecg_array.shape[1])
        return formatted

    except Exception as exc:
        return handle_error(exc, context="ECG")


# ============================================================
# TODO 2: SCREENING PPG
# ============================================================

def screen_ppg(signal: Any, fs: int = 125) -> Dict[str, Any]:
    """
    Thực hiện quy trình sàng lọc từ tín hiệu quang thể tích (PPG) và trích xuất đặc trưng nhịp tim.
    """
    try:
        ppg_array = _load_signal_from_source(signal)
        is_valid, msg = validate_input(ppg_array, expected_type=np.ndarray, min_length=30)
        if not is_valid:
            return handle_error(ValueError(msg), context="PPG")

        if np.isnan(ppg_array).any():
            ppg_array = np.nan_to_num(ppg_array, nan=0.0)

        extracted_features = _estimate_heart_rate_1d(ppg_array, fs=fs)

        predict_fn = _load_predict_function("PPG", ("predict_ppg", "predict", "run_prediction"))
        raw_result = predict_fn(ppg_array)

        formatted = format_prediction(raw_result, modality="PPG")
        formatted.setdefault("details", {})
        for k, v in extracted_features.items():
            formatted["details"].setdefault(k, v)
        formatted["details"]["signal_shape"] = list(ppg_array.shape)
        return formatted

    except Exception as exc:
        return handle_error(exc, context="PPG")


# ============================================================
# TODO 3: SCREENING CLINICAL
# ============================================================

def screen_clinical(input_data: Any) -> Dict[str, Any]:
    """
    Thực hiện sàng lọc từ dữ liệu lâm sàng (UCI Heart Disease).
    """
    try:
        if isinstance(input_data, dict):
            is_valid, msg = validate_input(
                input_data, expected_type=dict, required_keys=CLINICAL_FEATURES
            )
            if not is_valid:
                return handle_error(ValueError(msg), context="Clinical")

            ordered_values = [float(input_data[feat]) for feat in CLINICAL_FEATURES]
            feature_array = np.array([ordered_values], dtype=np.float32)
            recorded_inputs = {feat: float(input_data[feat]) for feat in CLINICAL_FEATURES}

        elif isinstance(input_data, pd.DataFrame):
            is_valid, msg = validate_input(input_data, expected_type=pd.DataFrame, min_length=1)
            if not is_valid:
                return handle_error(ValueError(msg), context="Clinical")

            if all(col in input_data.columns for col in CLINICAL_FEATURES):
                df_ordered = input_data[CLINICAL_FEATURES].iloc[[0]]
            elif input_data.shape[1] == len(CLINICAL_FEATURES):
                df_ordered = input_data.iloc[[0]]
            else:
                raise ValueError(
                    f"Dữ liệu Clinical cần đủ {len(CLINICAL_FEATURES)} đặc trưng: {CLINICAL_FEATURES}"
                )
            feature_array = df_ordered.to_numpy(dtype=np.float32)
            recorded_inputs = dict(zip(CLINICAL_FEATURES, feature_array[0].tolist()))

        elif isinstance(input_data, (list, tuple, np.ndarray)):
            arr = np.asarray(input_data, dtype=np.float32).flatten()
            if arr.size != len(CLINICAL_FEATURES):
                raise ValueError(
                    f"Yêu cầu đúng {len(CLINICAL_FEATURES)} đặc trưng lâm sàng, nhận được {arr.size}."
                )
            if np.isnan(arr).any() or np.isinf(arr).any():
                raise ValueError("Dữ liệu lâm sàng chứa giá trị không hợp lệ (NaN/Inf).")
            feature_array = arr.reshape(1, -1)
            recorded_inputs = dict(zip(CLINICAL_FEATURES, arr.tolist()))

        else:
            raise TypeError(
                "Dữ liệu Clinical phải là dict, DataFrame, list hoặc numpy.ndarray."
            )

        predict_fn = _load_predict_function(
            "Clinical", ("predict_clinical", "predict", "run_prediction")
        )
        raw_result = predict_fn(feature_array)

        formatted = format_prediction(raw_result, modality="Clinical")
        formatted.setdefault("details", {})
        formatted["details"]["input_features"] = recorded_inputs
        return formatted

    except Exception as exc:
        return handle_error(exc, context="Clinical")