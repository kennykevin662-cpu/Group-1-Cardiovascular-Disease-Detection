"""THÀNH VIÊN 4 – SYSTEM
Dashboard – Tổng hợp kết quả phân tích đa phương thức (ECG, PPG và Clinical).
Tính toán điểm rủi ro tim mạch toàn diện (Comprehensive Cardiovascular Risk Score).
"""
from typing import Any, Dict, Optional


# ============================================================
# TODO 1: QUẢN LÝ KẾT QUẢ ECG
# ============================================================

def get_ecg_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Trích xuất và định dạng kết quả ECG phục vụ hiển thị trên dashboard."""
    if not result or result.get("status") != "success":
        return {
            "available": False,
            "label": "Chưa có dữ liệu",
            "probability": 0.0,
            "is_abnormal": False,
            "heart_rate": None,
        }

    prob = result.get("probability", 0.0)
    pred = result.get("prediction", 0)
    return {
        "available": True,
        "label": result.get("label", "N/A"),
        "probability": prob,
        "is_abnormal": (pred == 1 or prob >= 0.5),
        "heart_rate": result.get("estimated_hr"),
        "rmssd": result.get("rmssd"),
    }


# ============================================================
# TODO 2: QUẢN LÝ KẾT QUẢ PPG
# ============================================================

def get_ppg_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Trích xuất và định dạng kết quả PPG phục vụ hiển thị trên dashboard."""
    if not result or result.get("status") != "success":
        return {
            "available": False,
            "label": "Chưa có dữ liệu",
            "heart_rate": None,
            "is_abnormal": False,
            "risk_level": "Chưa có dữ liệu",
        }

    hr = result.get("heart_rate")
    risk_level = result.get("risk_level", "Bình thường")
    is_abnormal = (risk_level != "Bình thường") or (hr is not None and (hr < 60 or hr > 100))

    return {
        "available": True,
        "label": result.get("label", "N/A"),
        "heart_rate": hr,
        "heart_rate_peaks": result.get("heart_rate_peaks"),
        "rmssd": result.get("rmssd"),
        "is_abnormal": is_abnormal,
        "risk_level": risk_level,
    }


# ============================================================
# TODO 3: QUẢN LÝ KẾT QUẢ CLINICAL
# ============================================================

def get_clinical_result(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Trích xuất và định dạng kết quả Clinical phục vụ hiển thị trên dashboard."""
    if not result or result.get("status") != "success":
        return {
            "available": False,
            "label": "Chưa có dữ liệu",
            "probability": 0.0,
            "is_diseased": False,
        }

    prob = result.get("probability", 0.0)
    pred = result.get("prediction", 0)
    return {
        "available": True,
        "label": result.get("label", "N/A"),
        "probability": prob,
        "is_diseased": (pred == 1 or prob >= 0.5),
        "details": result.get("details", {}),
    }


# ============================================================
# TODO 4: TỔNG HỢP KẾT QUẢ ĐA PHƯƠNG THỨC
# ============================================================

def get_dashboard_data(
    ecg: Optional[Dict[str, Any]] = None,
    ppg: Optional[Dict[str, Any]] = None,
    clinical: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tổng hợp kết quả của cả 3 phân hệ, tính toán điểm số rủi ro tim mạch tổng thể.
    
    Thuật toán tính điểm (Weighted Multi-modal Risk Index):
    - Trọng số Lâm sàng (Clinical): 40% (Đánh giá nguy cơ xơ vữa, tiền sử, mỡ máu, huyết áp)
    - Trọng số Điện tâm đồ (ECG): 40% (Phát hiện nhồi máu, dày thất, thiếu máu cơ tim)
    - Trọng số Quang thể tích (PPG): 20% (Đánh giá rối loạn nhịp tim nhanh/chậm, HRV)
    """
    ecg_info = get_ecg_result(ecg)
    ppg_info = get_ppg_result(ppg)
    clin_info = get_clinical_result(clinical)

    weights_sum = 0.0
    weighted_score = 0.0
    abnormal_count = 0
    total_available = 0

    # 1. Điểm từ Clinical
    if clin_info["available"]:
        w = 0.40
        weights_sum += w
        total_available += 1
        weighted_score += w * (clin_info["probability"] * 100.0)
        if clin_info["is_diseased"]:
            abnormal_count += 1

    # 2. Điểm từ ECG
    if ecg_info["available"]:
        w = 0.40
        weights_sum += w
        total_available += 1
        weighted_score += w * (ecg_info["probability"] * 100.0)
        if ecg_info["is_abnormal"]:
            abnormal_count += 1

    # 3. Điểm từ PPG
    if ppg_info["available"]:
        w = 0.20
        weights_sum += w
        total_available += 1
        # Nếu nhịp tim bất thường (<60 hoặc >100) -> 80 điểm, bình thường -> 20 điểm
        ppg_score = 80.0 if ppg_info["is_abnormal"] else 20.0
        weighted_score += w * ppg_score
        if ppg_info["is_abnormal"]:
            abnormal_count += 1

    # Tính điểm tổng thể
    if weights_sum > 0:
        overall_risk = round(weighted_score / weights_sum, 1)
    else:
        overall_risk = 0.0

    # Phân loại mức độ rủi ro
    if overall_risk >= 70.0:
        risk_category = "Nguy cơ rất cao (Very High Risk)"
        alert_color = "#dc3545"  # Đỏ
        advice = "Cần đến ngay cơ sở y tế chuyên khoa tim mạch để thực hiện xét nghiệm chuyên sâu và can thiệp kịp thời."
    elif overall_risk >= 50.0:
        risk_category = "Nguy cơ cao (High Risk)"
        alert_color = "#fd7e14"  # Cam đậm
        advice = "Hệ thống ghi nhận dấu hiệu bất thường. Khuyến nghị thăm khám bác sĩ tim mạch trong vòng 1-2 tuần."
    elif overall_risk >= 30.0:
        risk_category = "Nguy cơ trung bình (Moderate Risk)"
        alert_color = "#ffc107"  # Vàng
        advice = "Cần theo dõi định kỳ, điều chỉnh chế độ ăn uống ít muối, giảm mỡ và tăng cường vận động thể lực."
    else:
        risk_category = "Nguy cơ thấp (Low Risk / Healthy)"
        alert_color = "#28a745"  # Xanh lá
        advice = "Các chỉ số tim mạch nằm trong giới hạn an toàn. Duy trì lối sống lành mạnh và tầm soát hàng năm."

    return {
        "overall_risk_score": overall_risk,
        "risk_category": risk_category,
        "alert_color": alert_color,
        "advice": advice,
        "total_modalities_available": total_available,
        "abnormal_modalities_count": abnormal_count,
        "ecg": ecg_info,
        "ppg": ppg_info,
        "clinical": clin_info,
    }
