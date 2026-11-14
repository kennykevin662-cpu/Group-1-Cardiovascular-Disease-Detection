"""
THÀNH VIÊN 4 – SYSTEM
Dashboard – Tổng hợp kết quả phân tích ECG, PPG và Clinical.
"""

from typing import Any, Dict, List, Optional
from .hamhotro import format_prediction


def _normalize_modality_item(result: Any, modality: str) -> Optional[Dict[str, Any]]:
    """Chuẩn hóa kết quả của từng phân hệ trước khi hiển thị lên dashboard."""
    if result is None:
        return None
    if isinstance(result, dict) and result.get("status") == "error":
        return {
            "status": "error",
            "modality": modality,
            "prediction": None,
            "label": "Lỗi phân tích",
            "confidence": None,
            "confidence_pct": None,
            "risk_level": "Không xác định",
            "error_message": result.get("message", "Không thể xử lý dữ liệu."),
            "details": {},
        }

    formatted = format_prediction(result, modality=modality)
    conf = formatted.get("confidence")
    formatted["confidence_pct"] = round(conf * 100, 1) if conf is not None else None
    return formatted


# ============================================================
# TODO 1: QUẢN LÝ KẾT QUẢ ECG
# ============================================================

def get_ecg_result(result: Any) -> Optional[Dict[str, Any]]:
    """
    Xử lý kết quả ECG để hiển thị trên UI.
    """
    normalized = _normalize_modality_item(result, modality="ECG")
    if normalized is None:
        return None

    details = normalized.get("details", {})
    return {
        "modality": "ECG",
        "status": normalized["status"],
        "prediction": normalized["prediction"],
        "label": normalized["label"],
        "confidence": normalized["confidence"],
        "confidence_pct": normalized["confidence_pct"],
        "risk_level": normalized["risk_level"],
        "heart_rate": details.get("heart_rate"),
        "num_leads": details.get("num_leads"),
        "signal_shape": details.get("signal_shape"),
        "error_message": normalized.get("error_message"),
        "details": details,
    }


# ============================================================
# TODO 2: QUẢN LÝ KẾT QUẢ PPG
# ============================================================

def get_ppg_result(result: Any) -> Optional[Dict[str, Any]]:
    """
    Xử lý kết quả PPG để hiển thị trên UI (bao gồm nhịp tim heart_rate nếu có).
    """
    normalized = _normalize_modality_item(result, modality="PPG")
    if normalized is None:
        return None

    details = normalized.get("details", {})
    return {
        "modality": "PPG",
        "status": normalized["status"],
        "prediction": normalized["prediction"],
        "label": normalized["label"],
        "confidence": normalized["confidence"],
        "confidence_pct": normalized["confidence_pct"],
        "risk_level": normalized["risk_level"],
        "heart_rate": details.get("heart_rate"),
        "signal_shape": details.get("signal_shape"),
        "error_message": normalized.get("error_message"),
        "details": details,
    }


# ============================================================
# TODO 3: QUẢN LÝ KẾT QUẢ CLINICAL
# ============================================================

def get_clinical_result(result: Any) -> Optional[Dict[str, Any]]:
    """
    Xử lý kết quả Clinical để hiển thị trên UI.
    """
    normalized = _normalize_modality_item(result, modality="Clinical")
    if normalized is None:
        return None

    details = normalized.get("details", {})
    return {
        "modality": "Clinical",
        "status": normalized["status"],
        "prediction": normalized["prediction"],
        "label": normalized["label"],
        "confidence": normalized["confidence"],
        "confidence_pct": normalized["confidence_pct"],
        "risk_level": normalized["risk_level"],
        "input_features": details.get("input_features", {}),
        "error_message": normalized.get("error_message"),
        "details": details,
    }


# ============================================================
# TODO 4: TỔNG HỢP KẾT QUẢ
# ============================================================

def get_dashboard_data(
    ecg: Any = None,
    ppg: Any = None,
    clinical: Any = None,
) -> Dict[str, Any]:
    """
    Tổng hợp kết quả của 3 mô hình (ECG, PPG, Clinical) thành dữ liệu cho Dashboard.
    Hỗ trợ chạy đơn lẻ 1 mô hình hoặc kết hợp đa phương thức (2-3 mô hình).
    """
    ecg_data = get_ecg_result(ecg)
    ppg_data = get_ppg_result(ppg)
    clinical_data = get_clinical_result(clinical)

    modalities = {"ECG": ecg_data, "PPG": ppg_data, "Clinical": clinical_data}
    valid_results: List[Dict[str, Any]] = [
        item for item in modalities.values()
        if item is not None and item.get("status") == "success"
    ]

    if not valid_results:
        return {
            "has_results": False,
            "active_modalities": [],
            "overall_prediction": None,
            "overall_label": "Chưa có dữ liệu phân tích hợp lệ",
            "overall_confidence": None,
            "overall_confidence_pct": None,
            "overall_risk_level": "Chưa xác định",
            "recommendation": "Vui lòng nhập dữ liệu lâm sàng hoặc tải lên tín hiệu ECG/PPG để bắt đầu sàng lọc.",
            "modalities": modalities,
        }

    risk_scores: List[float] = []
    positive_count = 0

    for item in valid_results:
        pred = item["prediction"]
        conf = item["confidence"]
        if pred == 1:
            positive_count += 1
            risk_scores.append(conf if conf is not None else 0.75)
        else:
            risk_scores.append((1.0 - conf) if conf is not None else 0.25)

    avg_risk_score = float(sum(risk_scores) / len(risk_scores))
    overall_prediction = 1 if (avg_risk_score >= 0.5 or positive_count > len(valid_results) / 2) else 0

    if overall_prediction == 1:
        overall_conf = round(max(avg_risk_score, 0.51), 4)
        if avg_risk_score >= 0.75 or positive_count >= 2:
            overall_risk = "Cao"
            overall_label = "Nguy cơ cao mắc bệnh tim mạch"
            recommendation = (
                "Hệ thống phát hiện dấu hiệu bất thường trên các chỉ số tim mạch. "
                "Khuyến nghị người bệnh thăm khám bác sĩ chuyên khoa Tim mạch sớm để chẩn đoán chuyên sâu."
            )
        else:
            overall_risk = "Trung bình"
            overall_label = "Có dấu hiệu nghi ngờ bất thường tim mạch"
            recommendation = (
                "Phát hiện một số yếu tố nguy cơ. Nên theo dõi thêm huyết áp, nhịp tim "
                "và tham khảo ý kiến bác sĩ chuyên môn."
            )
    else:
        overall_conf = round(1.0 - avg_risk_score, 4)
        overall_risk = "Thấp"
        overall_label = "Các chỉ số sàng lọc trong giới hạn bình thường"
        recommendation = (
            "Chưa ghi nhận dấu hiệu bất thường rõ rệt. Duy trì lối sống lành mạnh "
            "và kiểm tra sức khỏe định kỳ."
        )

    return {
        "has_results": True,
        "active_modalities": [item["modality"] for item in valid_results],
        "positive_modalities_count": positive_count,
        "total_modalities_count": len(valid_results),
        "overall_prediction": overall_prediction,
        "overall_label": overall_label,
        "overall_confidence": overall_conf,
        "overall_confidence_pct": round(overall_conf * 100, 1),
        "overall_risk_level": overall_risk,
        "recommendation": recommendation,
        "modalities": modalities,
    }