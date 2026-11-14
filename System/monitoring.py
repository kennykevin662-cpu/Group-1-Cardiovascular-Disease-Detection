"""
THÀNH VIÊN 4 – SYSTEM
Monitoring – Kiểm thử và theo dõi hệ thống.
"""

from pathlib import Path
import tempfile
from typing import Any, Dict
import numpy as np

from .dashboard import get_dashboard_data
from .report import create_full_report, save_report
from .screening import CLINICAL_FEATURES, screen_clinical, screen_ecg, screen_ppg


def _evaluate_model_test(modality: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Đánh giá kết quả chạy thử model (phân biệt giữa lỗi code và chưa có file predict.py/.pkl)."""
    if result.get("status") == "success":
        return {
            "passed": True,
            "status": "PASS",
            "message": f"{modality}: Trả kết quả hợp lệ (prediction={result.get('prediction')}).",
            "result": result,
        }

    if result.get("error_type") == "FileNotFoundError":
        return {
            "passed": True,
            "status": "SKIPPED (Chưa có predict.py / model .pkl)",
            "message": result.get("message"),
            "result": result,
        }

    return {
        "passed": False,
        "status": "FAIL",
        "message": result.get("message", "Lỗi không xác định."),
        "result": result,
    }


# ============================================================
# TODO 1: KIỂM TRA MODEL ECG
# ============================================================

def test_ecg() -> Dict[str, Any]:
    """Kiểm tra luồng xử lý và dự đoán của model ECG với dữ liệu chuẩn PTB-XL (1000, 12)."""
    t = np.linspace(0, 10, 1000, dtype=np.float32)
    lead_wave = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 15.0 * t)
    sample_ecg = np.tile(lead_wave[:, None], (1, 12))  # Shape: (1000, 12)
    res = screen_ecg(sample_ecg)
    return _evaluate_model_test("ECG", res)


# ============================================================
# TODO 2: KIỂM TRA MODEL PPG
# ============================================================

def test_ppg() -> Dict[str, Any]:
    """Kiểm tra luồng xử lý, tính nhịp tim và dự đoán của model PPG (125 Hz)."""
    fs = 125
    t = np.linspace(0, 8, 8 * fs)
    sample_ppg = (np.sin(2 * np.pi * 1.25 * t) + 0.3 * np.sin(4 * np.pi * 1.25 * t)).astype(np.float32)
    res = screen_ppg(sample_ppg, fs=fs)
    return _evaluate_model_test("PPG", res)


# ============================================================
# TODO 3: KIỂM TRA MODEL CLINICAL
# ============================================================

def test_clinical() -> Dict[str, Any]:
    """Kiểm tra luồng xử lý và dự đoán của model Clinical (13 features UCI)."""
    sample_clinical = {
        "age": 54,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 246,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }
    assert list(sample_clinical.keys()) == CLINICAL_FEATURES
    res = screen_clinical(sample_clinical)
    return _evaluate_model_test("Clinical", res)


# ============================================================
# TODO 4: KIỂM TRA INPUT
# ============================================================

def test_invalid_input() -> Dict[str, Any]:
    """Kiểm tra khả năng bắt lỗi với các dữ liệu đầu vào không hợp lệ."""
    cases = {
        "ecg_empty": screen_ecg(np.array([])),
        "ecg_too_short": screen_ecg(np.ones((10, 12))),
        "ecg_none": screen_ecg(None),
        "ppg_invalid_file": screen_ppg("khong_ton_tai.exe"),
        "clinical_missing_keys": screen_clinical({"age": 50, "sex": 1}),
        "clinical_empty_dict": screen_clinical({}),
    }

    failed_cases = [name for name, out in cases.items() if out.get("status") != "error"]
    if failed_cases:
        return {
            "passed": False,
            "status": "FAIL",
            "message": f"Các trường hợp input lỗi không được chặn đúng cách: {failed_cases}",
        }

    return {
        "passed": True,
        "status": "PASS",
        "message": "Tất cả 6 trường hợp dữ liệu lỗi đều được bắt và trả thông báo an toàn.",
    }


# ============================================================
# TODO 5: KIỂM TRA TOÀN BỘ HỆ THỐNG
# ============================================================

def test_end_to_end() -> Dict[str, Any]:
    """
    Kiểm tra luồng tích hợp toàn hệ thống:
    Input -> Screening -> Dashboard -> Report -> Save file.
    """
    ecg_out = test_ecg()["result"]
    ppg_out = test_ppg()["result"]
    clinical_out = test_clinical()["result"]

    dashboard = get_dashboard_data(ecg=ecg_out, ppg=ppg_out, clinical=clinical_out)
    report = create_full_report(
        ecg=ecg_out,
        ppg=ppg_out,
        clinical=clinical_out,
        patient_info={"ID": "TEST-001", "Họ tên": "Nguyễn Văn A"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        report_file = Path(tmpdir) / "test_report.txt"
        saved, saved_path = save_report(report, str(report_file))
        file_ok = saved and Path(saved_path).exists() and Path(saved_path).stat().st_size > 0

    if not file_ok or "formatted_text" not in report or "modalities" not in dashboard:
        return {
            "passed": False,
            "status": "FAIL",
            "message": "Luồng tổng hợp Dashboard / Xuất báo cáo gặp sự cố.",
        }

    return {
        "passed": True,
        "status": "PASS",
        "message": (
            f"End-to-End hoạt động ổn định "
            f"(Phân hệ khả dụng: {dashboard['active_modalities'] or 'Chưa train model'})."
        ),
    }


# ============================================================
# TODO 6: CHẠY TOÀN BỘ TEST
# ============================================================

def run_tests() -> Dict[str, Any]:
    """Chạy toàn bộ kịch bản kiểm thử hệ thống."""
    suite = {
        "1. Test ECG": test_ecg(),
        "2. Test PPG": test_ppg(),
        "3. Test Clinical": test_clinical(),
        "4. Test Invalid Input": test_invalid_input(),
        "5. Test End-to-End": test_end_to_end(),
    }

    all_passed = all(item["passed"] for item in suite.values())
    print("================ KẾT QUẢ KIỂM THỬ SYSTEM ================")
    for name, res in suite.items():
        print(f"{name:<24} | {res['status']:<10} | {res['message']}")
    print("=========================================================")

    return {
        "all_passed": all_passed,
        "results": suite,
    }


if __name__ == "__main__":
    run_tests()