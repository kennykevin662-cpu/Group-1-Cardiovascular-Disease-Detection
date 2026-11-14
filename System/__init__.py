"""
SYSTEM PACKAGE

Phần System của hệ thống phát hiện bệnh tim mạch.
Phụ trách xử lý và điều phối ECG, PPG và Clinical.
"""

from .dashboard import (
    get_clinical_result,
    get_dashboard_data,
    get_ecg_result,
    get_ppg_result,
)
from .hamhotro import (
    format_prediction,
    handle_error,
    validate_file,
    validate_input,
)
from .monitoring import (
    run_tests,
    test_clinical,
    test_ecg,
    test_end_to_end,
    test_invalid_input,
    test_ppg,
)
from .report import (
    create_clinical_report,
    create_ecg_report,
    create_full_report,
    create_ppg_report,
    save_report,
)
from .screening import (
    CLINICAL_FEATURES,
    ECG_EXPECTED_LEADS,
    ECG_FS,
    screen_clinical,
    screen_ecg,
    screen_ppg,
)

__version__ = "1.0.0"

__all__ = [
    # Screening
    "CLINICAL_FEATURES",
    "ECG_FS",
    "ECG_EXPECTED_LEADS",
    "screen_ecg",
    "screen_ppg",
    "screen_clinical",
    # Dashboard
    "get_ecg_result",
    "get_ppg_result",
    "get_clinical_result",
    "get_dashboard_data",
    # Report
    "create_ecg_report",
    "create_ppg_report",
    "create_clinical_report",
    "create_full_report",
    "save_report",
    # Utils
    "validate_input",
    "validate_file",
    "format_prediction",
    "handle_error",
    # Monitoring
    "test_ecg",
    "test_ppg",
    "test_clinical",
    "test_invalid_input",
    "test_end_to_end",
    "run_tests",
]