"""THÀNH VIÊN 4 – SYSTEM
Utils – Các hàm hỗ trợ dùng chung (Validation, Formatting, Error Handling).
"""
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd

# Giới hạn sinh học y tế cho các đặc trưng lâm sàng
CLINICAL_BOUNDS = {
    "age": (1, 120, "Tuổi"),
    "trestbps": (50, 260, "Huyết áp lúc nghỉ (mmHg)"),
    "chol": (50, 600, "Cholesterol máu (mg/dl)"),
    "thalach": (40, 240, "Nhịp tim tối đa (bpm)"),
    "oldpeak": (0.0, 10.0, "Đoạn ST chênh (mm)"),
    "sex": ([0, 1], "Giới tính (0: Nữ, 1: Nam)"),
    "cp": ([1, 2, 3, 4], "Loại đau ngực (1-4)"),
    "fbs": ([0, 1], "Đường huyết lúc đói > 120mg/dl (0 hoặc 1)"),
    "restecg": ([0, 1, 2], "Điện tâm đồ lúc nghỉ (0, 1, 2)"),
    "exang": ([0, 1], "Đau thắt ngực khi gắng sức (0 hoặc 1)"),
    "slope": ([1, 2, 3], "Độ dốc đoạn ST (1, 2, 3)"),
    "ca": ([0, 1, 2, 3], "Số mạch máu lớn (0-3)"),
    "thal": ([3, 6, 7], "Tình trạng thalassaemia (3, 6, 7)"),
}


# ============================================================
# TODO 1: KIỂM TRA DỮ LIỆU
# ============================================================

def validate_input(data: Any, input_type: str = "clinical") -> Tuple[bool, str]:
    """Kiểm tra dữ liệu đầu vào có hợp lệ hay không.
    
    Args:
        data: Dữ liệu cần kiểm tra (dict với clinical, array với signal).
        input_type: Loại dữ liệu ('clinical', 'ecg', 'ppg').
        
    Returns:
        (is_valid, error_message): bool và chuỗi thông báo nếu có lỗi.
    """
    if data is None:
        return False, "Dữ liệu đầu vào không được để trống (None)."

    if input_type == "clinical":
        if isinstance(data, pd.DataFrame):
            if data.empty:
                return False, "Bảng dữ liệu lâm sàng rỗng."
            data = data.iloc[0].to_dict()

        if not isinstance(data, dict):
            return False, "Dữ liệu lâm sàng phải là dạng từ điển (Dictionary) hoặc DataFrame."

        # Kiểm tra từng trường và ngưỡng sinh học
        for feat, rule in CLINICAL_BOUNDS.items():
            if feat not in data or data[feat] is None or pd.isna(data[feat]):
                return False, f"Thiếu trường bắt buộc '{feat}' ({rule[-1]})."

            val = data[feat]
            try:
                val_num = float(val)
            except (ValueError, TypeError):
                return False, f"Trường '{feat}' phải là số, nhận được: {val}"

            if isinstance(rule[0], list):
                if int(val_num) not in rule[0]:
                    return False, f"Trường '{feat}' có giá trị {val_num} không hợp lệ. Phải thuộc {rule[0]}."
            elif isinstance(rule[0], (int, float)):
                min_v, max_v = rule[0], rule[1]
                if val_num < min_v or val_num > max_v:
                    return False, f"Trường '{feat}' ({rule[-1]}) có giá trị {val_num} nằm ngoài ngưỡng sinh học ({min_v} - {max_v})."

        return True, "Dữ liệu lâm sàng hợp lệ."

    elif input_type in ("ecg", "ppg"):
        try:
            arr = np.asarray(data, dtype=np.float32)
        except Exception as e:
            return False, f"Không thể chuyển tín hiệu {input_type.upper()} sang dạng số: {str(e)}"

        if arr.size == 0:
            return False, f"Tín hiệu {input_type.upper()} rỗng."

        if not np.all(np.isfinite(arr)):
            return False, f"Tín hiệu {input_type.upper()} chứa giá trị vô định (NaN hoặc Inf)."

        min_samples = 50 if input_type == "ecg" else 100
        if arr.shape[0] < min_samples:
            return False, f"Tín hiệu {input_type.upper()} quá ngắn ({arr.shape[0]} mẫu). Yêu cầu tối thiểu {min_samples} mẫu."

        return True, f"Tín hiệu {input_type.upper()} hợp lệ."

    return False, f"Không hỗ trợ kiểm tra loại dữ liệu '{input_type}'."


# ============================================================
# TODO 2: KIỂM TRA FILE
# ============================================================

def validate_file(file_path: Union[str, Path], allowed_exts: list = None) -> Tuple[bool, str]:
    """Kiểm tra tệp tin có tồn tại, đọc được và đúng định dạng.
    
    Args:
        file_path: Đường dẫn tệp tin.
        allowed_exts: Danh sách đuôi mở rộng cho phép, ví dụ ['.csv', '.npy', '.txt'].
        
    Returns:
        (is_valid, message): Kết quả kiểm tra và thông báo.
    """
    if not file_path:
        return False, "Đường dẫn tệp không được để trống."

    p = Path(file_path)
    if not p.exists():
        return False, f"Không tìm thấy tệp: {p.name}"

    if not p.is_file():
        return False, f"Đường dẫn không phải là tệp hợp lệ: {p.name}"

    if p.stat().st_size == 0:
        return False, f"Tệp tin rỗng (0 bytes): {p.name}"

    if allowed_exts:
        ext = p.suffix.lower()
        allowed = [e.lower() for e in allowed_exts]
        if ext not in allowed:
            return False, f"Định dạng tệp '{ext}' không được hỗ trợ. Chỉ chấp nhận: {', '.join(allowed)}"

    return True, f"Tệp {p.name} hợp lệ."


# ============================================================
# TODO 3: CHUẨN HÓA KẾT QUẢ
# ============================================================

def format_prediction(raw_result: Dict[str, Any], modality: str = "general") -> Dict[str, Any]:
    """Chuẩn hóa kết quả dự đoán từ các phân hệ khác nhau về một cấu trúc thống nhất.
    
    Args:
        raw_result: Kết quả thô từ model predict.
        modality: 'clinical', 'ecg', 'ppg', hoặc 'general'.
        
    Returns:
        Dict chuẩn hóa gồm status, modality, prediction, label, probability, confidence, details.
    """
    if not isinstance(raw_result, dict):
        return {
            "status": "error",
            "modality": modality,
            "message": "Kết quả dự đoán không phải dạng dictionary.",
        }

    status = raw_result.get("status", "success")
    pred = raw_result.get("prediction", 0)
    label = raw_result.get("label", "N/A")
    prob = raw_result.get("probability", 0.0)
    conf = raw_result.get("confidence", prob if prob >= 0.5 else 1.0 - prob)

    formatted = {
        "status": status,
        "modality": modality,
        "prediction": int(pred) if pred is not None else 0,
        "label": str(label),
        "probability": float(prob) if prob is not None else 0.0,
        "confidence": float(conf) if conf is not None else 0.0,
        "details": raw_result,
    }

    if modality == "ppg":
        hr = raw_result.get("heart_rate")
        hr_peaks = raw_result.get("heart_rate_peaks")
        formatted["heart_rate"] = hr
        formatted["heart_rate_peaks"] = hr_peaks
        formatted["rmssd"] = raw_result.get("rmssd")
        formatted["risk_level"] = raw_result.get("risk_level", "Bình thường")

    elif modality == "ecg":
        formatted["estimated_hr"] = raw_result.get("estimated_hr")
        formatted["rmssd"] = raw_result.get("rmssd")
        formatted["r_peaks_count"] = raw_result.get("r_peaks_count")

    return formatted


# ============================================================
# TODO 4: XỬ LÝ LỖI
# ============================================================

def handle_error(error: Exception, context: str = "") -> Dict[str, Any]:
    """Bắt lỗi, phân loại và tạo thông điệp giải thích thân thiện với người dùng và bác sĩ.
    
    Args:
        error: Đối tượng Exception.
        context: Ngữ cảnh xảy ra lỗi (ví dụ 'Dự đoán ECG', 'Đọc file PPG').
        
    Returns:
        Dict chứa mã lỗi, thông báo kỹ thuật và thông báo khuyến nghị.
    """
    err_type = type(error).__name__
    err_str = str(error)

    if isinstance(error, FileNotFoundError):
        user_msg = f"Hệ thống không tìm thấy tệp dữ liệu hoặc tệp mô hình yêu cầu trong {context}."
        advice = "Vui lòng kiểm tra lại đường dẫn tệp hoặc đảm bảo mô hình đã được huấn luyện đầy đủ."
    elif isinstance(error, ValueError):
        user_msg = f"Dữ liệu đầu vào hoặc giá trị tham số không phù hợp trong {context}."
        advice = "Vui lòng kiểm tra lại các trường thông tin hoặc kích thước ma trận tín hiệu."
    elif isinstance(error, KeyError):
        user_msg = f"Thiếu thông số bắt buộc trong dữ liệu ({err_str})."
        advice = "Vui lòng cung cấp đầy đủ các thuộc tính theo đúng quy định."
    elif isinstance(error, ModuleNotFoundError):
        user_msg = f"Thiếu thư viện phần mềm cần thiết: {err_str}."
        advice = "Hãy cài đặt thư viện bị thiếu thông qua pip."
    else:
        user_msg = f"Đã xảy ra lỗi không xác định trong quá trình {context}."
        advice = "Vui lòng liên hệ quản trị viên hệ thống để kiểm tra nhật ký ghi lỗi (logs)."

    return {
        "status": "error",
        "error_type": err_type,
        "context": context,
        "detail": err_str,
        "user_message": user_msg,
        "advice": advice,
    }
