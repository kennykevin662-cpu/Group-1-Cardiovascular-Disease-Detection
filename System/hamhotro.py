"""
THÀNH VIÊN 4 – SYSTEM
Utils – Các hàm hỗ trợ dùng chung.
"""

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd


# ============================================================
# TODO 1: KIỂM TRA DỮ LIỆU
# ============================================================

def validate_input(
    data: Any,
    expected_type: Optional[Union[type, Tuple[type, ...]]] = None,
    min_length: int = 1,
    required_keys: Optional[Iterable[str]] = None,
) -> Tuple[bool, str]:
    """
    Kiểm tra tính hợp lệ của dữ liệu đầu vào trước khi đưa vào mô hình.
    Trả về: (is_valid, message)
    """
    if data is None:
        return False, "Dữ liệu đầu vào bị trống (None)."

    if expected_type is not None and not isinstance(data, expected_type):
        return (
            False,
            f"Sai kiểu dữ liệu: yêu cầu {expected_type}, nhận được {type(data).__name__}.",
        )

    if isinstance(data, dict):
        if len(data) == 0:
            return False, "Từ điển dữ liệu rỗng."
        if required_keys:
            missing = [k for k in required_keys if k not in data or data[k] is None]
            if missing:
                return False, f"Thiếu các trường dữ liệu bắt buộc: {', '.join(missing)}."
        for k, v in data.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return False, f"Giá trị tại trường '{k}' không hợp lệ (NaN/Inf)."
        return True, "Dữ liệu hợp lệ."

    if isinstance(data, (pd.DataFrame, pd.Series)):
        if data.empty or len(data) < min_length:
            return False, f"Bảng dữ liệu rỗng hoặc ngắn hơn {min_length} mẫu."
        if data.isnull().all().all() if isinstance(data, pd.DataFrame) else data.isnull().all():
            return False, "Toàn bộ dữ liệu đều là giá trị thiếu (NaN)."
        return True, "Dữ liệu hợp lệ."

    if isinstance(data, np.ndarray):
        if data.size == 0 or data.shape[0] < min_length:
            return False, f"Mảng tín hiệu rỗng hoặc ngắn hơn mức tối thiểu ({min_length} mẫu)."
        if np.isnan(data).all() or np.isinf(data).any():
            return False, "Tín hiệu chứa giá trị không hợp lệ (toàn NaN hoặc có Inf)."
        return True, "Dữ liệu hợp lệ."

    if isinstance(data, (list, tuple, str)):
        if len(data) < min_length:
            return False, f"Độ dài dữ liệu không đủ (tối thiểu {min_length})."
        return True, "Dữ liệu hợp lệ."

    return True, "Dữ liệu hợp lệ."


# ============================================================
# TODO 2: KIỂM TRA FILE
# ============================================================

def validate_file(
    file: Any,
    allowed_extensions: Optional[Iterable[str]] = (".csv", ".npy", ".txt", ".hea", ".dat"),
) -> Tuple[bool, str]:
    """
    Kiểm tra file tải lên từ Streamlit (UploadedFile, list UploadedFile) hoặc đường dẫn cục bộ.
    Trả về: (is_valid, message)
    """
    if file is None:
        return False, "Chưa có file nào được cung cấp."

    # Hỗ trợ trường hợp upload đồng thời cặp file .hea và .dat của PTB-XL
    if isinstance(file, (list, tuple)):
        if len(file) == 0:
            return False, "Danh sách file trống."
        for f in file:
            ok, msg = validate_file(f, allowed_extensions=allowed_extensions)
            if not ok:
                return False, msg
        return True, "Các file hợp lệ."

    allowed = tuple(ext.lower() for ext in allowed_extensions) if allowed_extensions else ()

    if isinstance(file, (str, Path)):
        path = Path(file)
        if not path.exists() or not path.is_file():
            return False, f"Không tìm thấy file tại đường dẫn: {path}"
        if path.stat().st_size == 0:
            return False, "File rỗng (0 bytes)."
        if allowed and path.suffix.lower() not in allowed:
            return False, f"Định dạng '{path.suffix}' không được hỗ trợ. Chấp nhận: {', '.join(allowed)}"
        try:
            with open(path, "rb") as f:
                f.read(16)
        except Exception as exc:
            return False, f"Không thể đọc file: {exc}"
        return True, "File hợp lệ."

    filename = getattr(file, "name", "")
    if filename and allowed:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed:
            return False, f"Định dạng '{ext}' không được hỗ trợ. Chấp nhận: {', '.join(allowed)}"

    try:
        current_pos = file.tell() if hasattr(file, "tell") else 0
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(current_pos)
        if size == 0:
            return False, "File tải lên bị rỗng (0 bytes)."
    except Exception as exc:
        return False, f"Lỗi khi kiểm tra dung lượng file: {exc}"

    return True, "File hợp lệ."


# ============================================================
# TODO 3: CHUẨN HÓA KẾT QUẢ
# ============================================================

def format_prediction(prediction: Any, modality: str = "General") -> Dict[str, Any]:
    """
    Chuẩn hóa đầu ra từ các mô hình ECG, PPG, Clinical về cùng một cấu trúc thống nhất.
    """
    raw_pred = None
    confidence = None
    custom_label = None
    extra_details: Dict[str, Any] = {}

    if isinstance(prediction, dict):
        if prediction.get("status") == "error":
            return prediction

        raw_pred = (
            prediction.get("prediction")
            if "prediction" in prediction
            else prediction.get("pred", prediction.get("class", prediction.get("result")))
        )
        confidence = (
            prediction.get("confidence")
            if "confidence" in prediction
            else prediction.get("probability", prediction.get("prob", prediction.get("score")))
        )
        custom_label = prediction.get("label") or prediction.get("conclusion")

        reserved_keys = {
            "status", "modality", "prediction", "pred", "class", "result",
            "confidence", "probability", "prob", "score", "label", "conclusion"
        }
        extra_details = {k: v for k, v in prediction.items() if k not in reserved_keys}

    elif isinstance(prediction, (tuple, list)) and len(prediction) >= 1:
        raw_pred = prediction[0]
        if len(prediction) >= 2:
            confidence = prediction[1]

    elif isinstance(prediction, np.ndarray):
        flat = prediction.flatten()
        if flat.size == 1:
            val = float(flat[0])
            if 0.0 < val < 1.0:
                confidence = val
                raw_pred = 1 if val >= 0.5 else 0
            else:
                raw_pred = int(round(val))
        elif flat.size >= 2:
            raw_pred = int(np.argmax(flat))
            confidence = float(np.max(flat))
    else:
        raw_pred = prediction

    if isinstance(raw_pred, (np.generic,)):
        raw_pred = raw_pred.item()

    if isinstance(raw_pred, str):
        lowered = raw_pred.strip().lower()
        if lowered in {"1", "abnormal", "disease", "positive", "high", "bệnh", "bất thường", "mi", "sttc", "cd", "hyp"}:
            norm_pred = 1
        elif lowered in {"0", "normal", "healthy", "negative", "low", "bình thường", "norm"}:
            norm_pred = 0
        else:
            norm_pred = 1
            custom_label = custom_label or raw_pred
    elif isinstance(raw_pred, (int, float, bool)):
        if isinstance(raw_pred, float) and 0.0 < raw_pred < 1.0 and confidence is None:
            confidence = float(raw_pred)
            norm_pred = 1 if raw_pred >= 0.5 else 0
        else:
            norm_pred = int(1 if raw_pred >= 1 else 0)
    else:
        norm_pred = 0

    norm_conf: Optional[float] = None
    if confidence is not None:
        if isinstance(confidence, (np.generic,)):
            confidence = confidence.item()
        try:
            c_val = float(confidence)
            if 1.0 < c_val <= 100.0:
                c_val = c_val / 100.0
            norm_conf = round(max(0.0, min(1.0, c_val)), 4)
        except (TypeError, ValueError):
            norm_conf = None

    if custom_label:
        label = str(custom_label)
    else:
        label = (
            "Phát hiện dấu hiệu bất thường / Nguy cơ tim mạch"
            if norm_pred == 1
            else "Bình thường (NORM) / Chưa phát hiện bất thường"
        )

    if norm_pred == 0:
        risk_level = "Thấp"
    elif norm_conf is not None and norm_conf < 0.7:
        risk_level = "Trung bình"
    else:
        risk_level = "Cao"

    return {
        "status": "success",
        "modality": modality,
        "prediction": norm_pred,
        "label": label,
        "confidence": norm_conf,
        "risk_level": risk_level,
        "details": extra_details,
    }


# ============================================================
# TODO 4: XỬ LÝ LỖI
# ============================================================

def handle_error(error: Exception, context: str = "") -> Dict[str, Any]:
    """
    Phân loại ngoại lệ và tạo thông báo thân thiện hiển thị lên giao diện Streamlit.
    """
    if isinstance(error, FileNotFoundError):
        error_type = "FileNotFoundError"
        user_msg = "Không tìm thấy tệp dữ liệu hoặc tệp mô hình đã huấn luyện (.pkl)."
    elif isinstance(error, (ImportError, ModuleNotFoundError)):
        error_type = "ImportError"
        user_msg = "Không thể nạp module dự đoán của phân hệ. Vui lòng kiểm tra thư mục mohinh/."
    elif isinstance(error, ValueError):
        error_type = "ValueError"
        user_msg = f"Dữ liệu đầu vào không đúng định dạng hoặc kích thước: {error}"
    elif isinstance(error, TypeError):
        error_type = "TypeError"
        user_msg = f"Kiểu dữ liệu không tương thích khi xử lý: {error}"
    elif isinstance(error, KeyError):
        error_type = "KeyError"
        user_msg = f"Thiếu trường thông tin cần thiết: {error}"
    else:
        error_type = type(error).__name__
        user_msg = f"Đã xảy ra lỗi trong quá trình xử lý: {error}"

    prefix = f"[{context}] " if context else ""
    return {
        "status": "error",
        "modality": context or "System",
        "error_type": error_type,
        "message": f"{prefix}{user_msg}",
        "detail": str(error),
    }