"""
THÀNH VIÊN 4 – SYSTEM
Report – Tạo và quản lý báo cáo kết quả.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from .dashboard import (
    get_clinical_result,
    get_dashboard_data,
    get_ecg_result,
    get_ppg_result,
)


def _format_confidence_text(conf_pct: Optional[float]) -> str:
    return f"{conf_pct:.1f}%" if conf_pct is not None else "Không có thông tin"


# ============================================================
# TODO 1: BÁO CÁO ECG
# ============================================================

def create_ecg_report(result: Any) -> Dict[str, Any]:
    """
    Tạo báo cáo chi tiết cho phân hệ ECG.
    """
    data = get_ecg_result(result)
    if data is None:
        return {"available": False, "summary": "Không có dữ liệu phân tích ECG."}

    if data["status"] == "error":
        return {
            "available": False,
            "error": data.get("error_message"),
            "summary": f"Phân tích ECG gặp lỗi: {data.get('error_message')}",
        }

    hr_line = f"\n- Nhịp tim tham khảo (Lead II): {data['heart_rate']} BPM" if data.get("heart_rate") else ""
    summary = (
        f"- Kết luận ECG: {data['label']} (Mã dự đoán: {data['prediction']})\n"
        f"- Độ tin cậy: {_format_confidence_text(data['confidence_pct'])}\n"
        f"- Mức độ nguy cơ: {data['risk_level']}\n"
        f"- Kích thước tín hiệu đầu vào: {data.get('signal_shape') or 'N/A'}"
        f"{hr_line}"
    )
    return {
        "available": True,
        "modality": "ECG",
        "prediction": data["prediction"],
        "label": data["label"],
        "confidence": data["confidence"],
        "confidence_pct": data["confidence_pct"],
        "risk_level": data["risk_level"],
        "heart_rate": data.get("heart_rate"),
        "signal_shape": data.get("signal_shape"),
        "summary": summary,
    }


# ============================================================
# TODO 2: BÁO CÁO PPG
# ============================================================

def create_ppg_report(result: Any) -> Dict[str, Any]:
    """
    Tạo báo cáo chi tiết cho phân hệ PPG (kèm nhịp tim ước tính).
    """
    data = get_ppg_result(result)
    if data is None:
        return {"available": False, "summary": "Không có dữ liệu phân tích PPG."}

    if data["status"] == "error":
        return {
            "available": False,
            "error": data.get("error_message"),
            "summary": f"Phân tích PPG gặp lỗi: {data.get('error_message')}",
        }

    hr_text = f"{data['heart_rate']} BPM" if data.get("heart_rate") is not None else "Chưa xác định"
    summary = (
        f"- Kết luận PPG: {data['label']} (Mã dự đoán: {data['prediction']})\n"
        f"- Độ tin cậy: {_format_confidence_text(data['confidence_pct'])}\n"
        f"- Mức độ nguy cơ: {data['risk_level']}\n"
        f"- Nhịp tim ước tính (Heart Rate): {hr_text}\n"
        f"- Kích thước tín hiệu đầu vào: {data.get('signal_shape') or 'N/A'}"
    )
    return {
        "available": True,
        "modality": "PPG",
        "prediction": data["prediction"],
        "label": data["label"],
        "confidence": data["confidence"],
        "confidence_pct": data["confidence_pct"],
        "risk_level": data["risk_level"],
        "heart_rate": data.get("heart_rate"),
        "details": data.get("details", {}),
        "summary": summary,
    }


# ============================================================
# TODO 3: BÁO CÁO CLINICAL
# ============================================================

def create_clinical_report(result: Any) -> Dict[str, Any]:
    """
    Tạo báo cáo chi tiết cho phân hệ Clinical.
    """
    data = get_clinical_result(result)
    if data is None:
        return {"available": False, "summary": "Không có dữ liệu phân tích lâm sàng (Clinical)."}

    if data["status"] == "error":
        return {
            "available": False,
            "error": data.get("error_message"),
            "summary": f"Phân tích Clinical gặp lỗi: {data.get('error_message')}",
        }

    features = data.get("input_features", {})
    key_info = ""
    if features:
        key_info = (
            f"\n- Chỉ số nổi bật: Tuổi={features.get('age')}, "
            f"Huyết áp nghỉ={features.get('trestbps')} mmHg, "
            f"Cholesterol={features.get('chol')} mg/dl, "
            f"Nhịp tim tối đa={features.get('thalach')} bpm"
        )

    summary = (
        f"- Kết luận lâm sàng: {data['label']} (Mã dự đoán: {data['prediction']})\n"
        f"- Độ tin cậy: {_format_confidence_text(data['confidence_pct'])}\n"
        f"- Mức độ nguy cơ: {data['risk_level']}"
        f"{key_info}"
    )
    return {
        "available": True,
        "modality": "Clinical",
        "prediction": data["prediction"],
        "label": data["label"],
        "confidence": data["confidence"],
        "confidence_pct": data["confidence_pct"],
        "risk_level": data["risk_level"],
        "input_features": features,
        "summary": summary,
    }


# ============================================================
# TODO 4: TỔNG HỢP BÁO CÁO
# ============================================================

def create_full_report(
    ecg: Any = None,
    ppg: Any = None,
    clinical: Any = None,
    patient_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tổng hợp báo cáo toàn diện từ 3 phân hệ ECG, PPG và Clinical.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ecg_rep = create_ecg_report(ecg)
    ppg_rep = create_ppg_report(ppg)
    clinical_rep = create_clinical_report(clinical)
    dashboard = get_dashboard_data(ecg=ecg, ppg=ppg, clinical=clinical)

    lines = [
        "============================================================",
        "BÁO CÁO SÀNG LỌC NGUY CƠ BỆNH TIM MẠCH ĐA PHƯƠNG THỨC",
        "============================================================",
        f"Thời gian tạo báo cáo: {timestamp}",
    ]

    if patient_info:
        lines.append("\n[THÔNG TIN ĐỐI TƯỢNG SÀNG LỌC]")
        for k, v in patient_info.items():
            lines.append(f"- {k}: {v}")

    lines.extend([
        "\n[1. KẾT QUẢ TỔNG HỢP]",
        f"- Phân hệ đã phân tích: {', '.join(dashboard['active_modalities']) or 'Không có'}",
        f"- Kết luận chung: {dashboard['overall_label']}",
        f"- Mức độ nguy cơ tổng hợp: {dashboard['overall_risk_level']}",
        f"- Độ tin cậy tổng hợp: {_format_confidence_text(dashboard['overall_confidence_pct'])}",
        f"- Khuyến nghị: {dashboard['recommendation']}",
        "\n[2. CHI TIẾT TỪNG PHÂN HỆ]",
        "--- Phân hệ ECG ---",
        ecg_rep["summary"],
        "\n--- Phân hệ PPG ---",
        ppg_rep["summary"],
        "\n--- Phân hệ Lâm sàng (Clinical) ---",
        clinical_rep["summary"],
        "\n------------------------------------------------------------",
        "Lưu ý: Kết quả từ hệ thống AI chỉ mang tính chất hỗ trợ sàng lọc,",
        "không thay thế chẩn đoán chính thức của bác sĩ chuyên khoa.",
    ])

    formatted_text = "\n".join(lines)

    return {
        "timestamp": timestamp,
        "patient_info": patient_info or {},
        "overall": {
            "has_results": dashboard["has_results"],
            "active_modalities": dashboard["active_modalities"],
            "prediction": dashboard["overall_prediction"],
            "label": dashboard["overall_label"],
            "confidence": dashboard["overall_confidence"],
            "confidence_pct": dashboard["overall_confidence_pct"],
            "risk_level": dashboard["overall_risk_level"],
            "recommendation": dashboard["recommendation"],
        },
        "ecg": ecg_rep,
        "ppg": ppg_rep,
        "clinical": clinical_rep,
        "formatted_text": formatted_text,
    }


# ============================================================
# TODO 5: LƯU BÁO CÁO
# ============================================================

def save_report(report: Any, filename: str) -> Tuple[bool, str]:
    """
    Lưu báo cáo ra tệp (.json, .txt hoặc .md).
    Trả về: (success, filepath_or_error_message)
    """
    try:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix.lower() == ".json":
            payload = report if isinstance(report, dict) else {"content": str(report)}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            if isinstance(report, dict):
                text_content = report.get("formatted_text") or json.dumps(
                    report, ensure_ascii=False, indent=2
                )
            else:
                text_content = str(report)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text_content)

        return True, str(out_path.resolve())
    except Exception as exc:
        return False, f"Không thể lưu báo cáo: {exc}"