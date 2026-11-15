"""THÀNH VIÊN 4 – SYSTEM
Screening – Logic sàng lọc bệnh tim mạch.

Xử lý dữ liệu đầu vào và gọi mô hình tương ứng:
ECG / PPG / Clinical.
"""
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from .hamhotro import format_prediction, handle_error, validate_file, validate_input

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


# ============================================================
# TODO 1: SCREENING ECG
# ============================================================

def screen_ecg(signal_or_path: Union[np.ndarray, str, Path], fs: int = 100) -> Dict[str, Any]:
    """Thực hiện sàng lọc bất thường tim mạch từ tín hiệu điện tâm đồ ECG.
    
    Args:
        signal_or_path: Mảng numpy hoặc đường dẫn tới tệp ECG (.npy, .csv, WFDB record).
        fs: Tần số lấy mẫu (mặc định 100 Hz theo chuẩn PTB-XL records100).
        
    Returns:
        Dict: Kết quả sàng lọc đã được chuẩn hóa.
    """
    try:
        from mohinh.ECG.predict import predict as ecg_predict, predict_record

        # 1. Nếu là đường dẫn tệp
        if isinstance(signal_or_path, (str, Path)):
            is_valid, msg = validate_file(signal_or_path)
            if not is_valid:
                return {"status": "error", "modality": "ecg", "message": msg}
            raw = predict_record(str(signal_or_path), fs=fs)
        # 2. Nếu là mảng numpy
        else:
            is_valid, msg = validate_input(signal_or_path, input_type="ecg")
            if not is_valid:
                return {"status": "error", "modality": "ecg", "message": msg}
            raw = ecg_predict(signal_or_path, fs=fs)

        return format_prediction(raw, modality="ecg")

    except Exception as e:
        return handle_error(e, context="Sàng lọc ECG")


# ============================================================
# TODO 2: SCREENING PPG
# ============================================================

def screen_ppg(signal_or_path: Union[np.ndarray, str, Path], fs: int = 1000) -> Dict[str, Any]:
    """Thực hiện sàng lọc nhịp mạch & rủi ro từ tín hiệu quang thể tích PPG.
    
    Args:
        signal_or_path: Mảng numpy 1D hoặc đường dẫn tệp PPG (.txt, .csv, .npy).
        fs: Tần số lấy mẫu (mặc định 1000 Hz theo chuẩn PPG-BP).
        
    Returns:
        Dict: Kết quả phân tích nhịp tim và rủi ro PPG.
    """
    try:
        from mohinh.PPG.predict import predict as ppg_predict, predict_file

        # 1. Nếu là đường dẫn tệp
        if isinstance(signal_or_path, (str, Path)):
            is_valid, msg = validate_file(signal_or_path, allowed_exts=[".txt", ".csv", ".npy"])
            if not is_valid:
                return {"status": "error", "modality": "ppg", "message": msg}
            raw = predict_file(str(signal_or_path), fs=fs)
        # 2. Nếu là mảng tín hiệu
        else:
            is_valid, msg = validate_input(signal_or_path, input_type="ppg")
            if not is_valid:
                return {"status": "error", "modality": "ppg", "message": msg}
            raw = ppg_predict(signal_or_path, fs=fs)

        return format_prediction(raw, modality="ppg")

    except Exception as e:
        return handle_error(e, context="Sàng lọc PPG")


# ============================================================
# TODO 3: SCREENING CLINICAL
# ============================================================

def screen_clinical(input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """Thực hiện phân tích nguy cơ bệnh tim từ 13 chỉ số lâm sàng.
    
    Args:
        input_data: Dictionary hoặc DataFrame chứa 13 thuộc tính:
                    age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                    exang, oldpeak, slope, ca, thal.
                    
    Returns:
        Dict: Kết quả dự đoán nhãn (Có bệnh / Không bệnh) và xác suất nguy cơ.
    """
    try:
        is_valid, msg = validate_input(input_data, input_type="clinical")
        if not is_valid:
            return {"status": "error", "modality": "clinical", "message": msg}

        from mohinh.Clinical.predict import predict_dict, predict_df

        if isinstance(input_data, pd.DataFrame):
            df_res = predict_df(input_data)
            row = df_res.iloc[0]
            raw = {
                "prediction": int(row["probability"] >= 0.5),
                "label": str(row["label"]),
                "probability": float(row["probability"]),
                "details": row.to_dict(),
            }
        else:
            raw = predict_dict(input_data)

        return format_prediction(raw, modality="clinical")

    except Exception as e:
        return handle_error(e, context="Sàng lọc Dữ liệu Lâm sàng")
