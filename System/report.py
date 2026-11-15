"""THÀNH VIÊN 4 – SYSTEM
Report – Tạo và quản lý báo cáo kết quả sàng lọc tim mạch.
Xuất báo cáo định dạng Markdown / Văn bản chuẩn y khoa.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .dashboard import get_dashboard_data


# ============================================================
# TODO 1: BÁO CÁO ECG
# ============================================================

def create_ecg_report(result: Dict[str, Any]) -> str:
    """Tạo báo cáo chi tiết cho phân hệ ECG."""
    if not result or result.get("status") != "success":
        return "### Báo cáo ECG: Chưa có dữ liệu hoặc phân tích thất bại.\n"

    prob = result.get("probability", 0.0)
    label = result.get("label", "N/A")
    hr = result.get("estimated_hr", "Không xác định")
    rmssd = result.get("rmssd", "Không xác định")
    r_peaks = result.get("r_peaks_count", "N/A")

    return f"""### 🫀 1. Phân tích Điện tâm đồ (ECG - 12 Leads)
- **Kết luận mô hình**: **{label}**
- **Xác suất bất thường**: **{prob:.2%}**
- **Nhịp tim ước tính (Lead II)**: {hr} bpm
- **Biến thiên nhịp tim (RMSSD)**: {rmssd}
- **Số đỉnh sóng R phát hiện**: {r_peaks}
- **Đánh giá**: {'Cần chú ý bất thường về dẫn truyền, phì đại hoặc biến đổi đoạn ST.' if prob >= 0.5 else 'Dẫn truyền và hình thái sóng điện tim trong giới hạn bình thường.'}
"""


# ============================================================
# TODO 2: BÁO CÁO PPG
# ============================================================

def create_ppg_report(result: Dict[str, Any]) -> str:
    """Tạo báo cáo chi tiết cho phân hệ PPG."""
    if not result or result.get("status") != "success":
        return "### Báo cáo PPG: Chưa có dữ liệu hoặc phân tích thất bại.\n"

    hr = result.get("heart_rate", "N/A")
    hr_peaks = result.get("heart_rate_peaks", "N/A")
    label = result.get("label", "N/A")
    rmssd = result.get("rmssd", "N/A")

    return f"""### 📈 2. Phân tích Quang thể tích (PPG)
- **Nhịp tim mô hình ước lượng**: **{hr} bpm**
- **Nhịp tim tính từ đỉnh xung mạch**: **{hr_peaks} bpm**
- **Trạng thái nhịp**: **{label}**
- **Chỉ số biến thiên nhịp tim (HRV - RMSSD)**: {rmssd}
- **Đánh giá**: {'Phát hiện dấu hiệu rối loạn nhịp tim (nhanh hoặc chậm bất thường).' if label != 'Nhịp tim bình thường (Normal Sinus Rhythm)' else 'Nhịp đập tuần hoàn ổn định.'}
"""


# ============================================================
# TODO 3: BÁO CÁO CLINICAL
# ============================================================

def create_clinical_report(result: Dict[str, Any]) -> str:
    """Tạo báo cáo chi tiết cho phân hệ Dữ liệu Lâm sàng."""
    if not result or result.get("status") != "success":
        return "### Báo cáo Lâm sàng: Chưa có dữ liệu hoặc phân tích thất bại.\n"

    prob = result.get("probability", 0.0)
    label = result.get("label", "N/A")
    details = result.get("details", {})

    items_str = ""
    if details:
        items_str = "\n- **Thông số chính**: " + ", ".join([f"{k}: {v}" for k, v in list(details.items())[:6]])

    return f"""### 📋 3. Phân tích Dữ liệu Lâm sàng (Clinical)
- **Kết luận rủi ro**: **{label}**
- **Xác suất có bệnh tim**: **{prob:.2%}**{items_str}
- **Đánh giá**: {'Có nguy cơ cao mắc bệnh mạch vành dựa trên các yếu tố nguy cơ.' if prob >= 0.5 else 'Các chỉ số sinh hóa và triệu chứng không biểu hiện nguy cơ tim mạch nghiêm trọng.'}
"""


# ============================================================
# TODO 4: TỔNG HỢP BÁO CÁO
# ============================================================

def create_full_report(
    ecg: Optional[Dict[str, Any]] = None,
    ppg: Optional[Dict[str, Any]] = None,
    clinical: Optional[Dict[str, Any]] = None,
    patient_info: Optional[Dict[str, str]] = None,
) -> str:
    """Tạo phiếu kết quả sàng lọc tim mạch toàn diện."""
    dash = get_dashboard_data(ecg=ecg, ppg=ppg, clinical=clinical)
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    p_name = patient_info.get("name", "Bệnh nhân ẩn danh") if patient_info else "Bệnh nhân ẩn danh"
    p_id = patient_info.get("id", "BN-" + datetime.now().strftime("%y%m%d%H%M")) if patient_info else "BN-DEMO"
    p_age = patient_info.get("age", "N/A") if patient_info else "N/A"
    p_sex = patient_info.get("sex", "N/A") if patient_info else "N/A"

    report_lines = [
        "# PHIẾU KẾT QUẢ SÀNG LỌC NGUY CƠ BỆNH TIM MẠCH",
        "## Hệ thống AI đa phương thức (ECG + PPG + Clinical)",
        "---",
        f"**Thời gian thực hiện**: {now_str}  ",
        f"**Mã hồ sơ**: `{p_id}` | **Họ và tên**: **{p_name}** | **Tuổi**: {p_age} | **Giới tính**: {p_sex}",
        "---",
        "## I. KẾT LUẬN TỔNG THỂ",
        f"- **Điểm rủi ro tim mạch toàn diện (CRI)**: **{dash['overall_risk_score']}/100**",
        f"- **Phân loại mức độ nguy cơ**: **{dash['risk_category']}**",
        f"- **Số nguồn dữ liệu bất thường**: {dash['abnormal_modalities_count']}/{dash['total_modalities_available']} phân hệ",
        f"- **Khuyến nghị chuyên khoa**: *{dash['advice']}*",
        "---",
        "## II. KẾT QUẢ TỪNG PHÂN HỆ",
        create_ecg_report(ecg),
        create_ppg_report(ppg),
        create_clinical_report(clinical),
        "---",
        "## III. LƯU Ý Y KHOA",
        "> Kết quả trên được phân tích tự động bởi hệ thống AI hỗ trợ ra quyết định lâm sàng.",
        "> Phiếu kết quả này không thay thế chẩn đoán xác định của bác sĩ chuyên khoa tim mạch.",
    ]

    return "\n".join(report_lines)


# ============================================================
# TODO 5: LƯU BÁO CÁO
# ============================================================

def save_report(report_text: str, filename: str = "report.md") -> str:
    """Lưu nội dung báo cáo thành tệp."""
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(report_text)
    return str(p.resolve())
