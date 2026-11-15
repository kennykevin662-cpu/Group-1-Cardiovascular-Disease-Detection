"""SYSTEM PACKAGE - THÀNH VIÊN 4

Lớp điều phối trung tâm của Hệ thống AI phát hiện bệnh tim mạch đa phương thức.
Tích hợp và kết nối 3 phân hệ:
- ECG (Điện tâm đồ)
- PPG (Quang thể tích)
- Clinical (Dữ liệu lâm sàng)
"""

from .hamhotro import format_prediction, handle_error, validate_file, validate_input
from .screening import screen_clinical, screen_ecg, screen_ppg
from .dashboard import get_dashboard_data, get_ecg_result, get_ppg_result, get_clinical_result
from .report import create_clinical_report, create_ecg_report, create_full_report, create_ppg_report, save_report
from .monitoring import run_tests

__all__ = [
    "validate_input",
    "validate_file",
    "format_prediction",
    "handle_error",
    "screen_clinical",
    "screen_ecg",
    "screen_ppg",
    "get_dashboard_data",
    "get_ecg_result",
    "get_ppg_result",
    "get_clinical_result",
    "create_clinical_report",
    "create_ecg_report",
    "create_ppg_report",
    "create_full_report",
    "save_report",
    "run_tests",
]
