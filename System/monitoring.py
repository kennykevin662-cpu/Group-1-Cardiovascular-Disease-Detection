"""THÀNH VIÊN 4 – SYSTEM
Monitoring – Kiểm thử và theo dõi toàn bộ hệ thống (Unit Tests & Integration Tests).
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# Đảm bảo console Windows không bị UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Đảm bảo PYTHONPATH có thư mục gốc dự án
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from System.dashboard import get_dashboard_data
from System.hamhotro import validate_file, validate_input
from System.report import create_full_report
from System.screening import screen_clinical, screen_ecg, screen_ppg


# ============================================================
# TODO 1: KIỂM TRA MODEL ECG
# ============================================================

def test_ecg() -> bool:
    print("[TEST 1/5] Kiểm thử mô hình ECG...")
    try:
        # Chuẩn bị tín hiệu ECG mẫu 12 kênh (1000 mẫu, tần số 100 Hz = 10 giây)
        dummy_ecg = np.random.randn(1000, 12).astype(np.float32)
        res = screen_ecg(dummy_ecg, fs=100)
        assert res["status"] == "success", f"ECG trả về status={res.get('status')}: {res.get('message')}"
        assert "probability" in res, "Thiếu trường 'probability' trong kết quả ECG"
        assert "label" in res, "Thiếu trường 'label' trong kết quả ECG"
        print(f"  ✓ Thành công: Label='{res['label']}', Xác suất={res['probability']:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Thất bại: {e}")
        traceback.print_exc()
        return False


# ============================================================
# TODO 2: KIỂM TRA MODEL PPG
# ============================================================

def test_ppg() -> bool:
    print("[TEST 2/5] Kiểm thử mô hình PPG...")
    try:
        # Chuẩn bị tín hiệu PPG mẫu 1 kênh (2100 mẫu)
        # Tạo sóng sin chu kỳ tương tự mạch đập + nhiễu
        t = np.linspace(0, 2.1, 2100)
        dummy_ppg = 2000.0 + 300.0 * np.sin(2 * np.pi * 1.2 * t) + np.random.randn(2100) * 10
        res = screen_ppg(dummy_ppg, fs=1000)
        assert res["status"] == "success", f"PPG trả về status={res.get('status')}: {res.get('message')}"
        assert "heart_rate" in res, "Thiếu trường 'heart_rate' trong kết quả PPG"
        assert res["heart_rate"] is not None, "Giá trị nhịp tim bị None"
        print(f"  ✓ Thành công: Nhịp tim dự đoán={res['heart_rate']} bpm, Trạng thái='{res['label']}'")
        return True
    except Exception as e:
        print(f"  ✗ Thất bại: {e}")
        traceback.print_exc()
        return False


# ============================================================
# TODO 3: KIỂM TRA MODEL CLINICAL
# ============================================================

def test_clinical() -> bool:
    print("[TEST 3/5] Kiểm thử mô hình Clinical...")
    try:
        sample_patient = {
            "age": 58,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 240,
            "fbs": 0,
            "restecg": 0,
            "thalach": 160,
            "exang": 0,
            "oldpeak": 1.2,
            "slope": 1,
            "ca": 0,
            "thal": 3,
        }
        res = screen_clinical(sample_patient)
        assert res["status"] == "success", f"Clinical trả về status={res.get('status')}: {res.get('message')}"
        assert "probability" in res, "Thiếu trường 'probability' trong kết quả Clinical"
        print(f"  ✓ Thành công: Label='{res['label']}', Xác suất có bệnh={res['probability']:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Thất bại: {e}")
        traceback.print_exc()
        return False


# ============================================================
# TODO 4: KIỂM TRA DỮ LIỆU ĐẦU VÀO KHÔNG HỢP LỆ
# ============================================================

def test_invalid_input() -> bool:
    print("[TEST 4/5] Kiểm thử bắt lỗi đầu vào bất hợp lệ...")
    passed_cases = 0

    # 1. Tín hiệu rỗng
    v1, _ = validate_input(np.array([]), input_type="ecg")
    if not v1:
        passed_cases += 1

    # 2. Thiếu trường bắt buộc trong Clinical
    v2, _ = validate_input({"age": 50, "sex": 1}, input_type="clinical")
    if not v2:
        passed_cases += 1

    # 3. Giá trị ngoài giới hạn sinh học (ví dụ: tuổi 250)
    v3, _ = validate_input({"age": 250, "sex": 1, "cp": 1, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3}, input_type="clinical")
    if not v3:
        passed_cases += 1

    # 4. File không tồn tại
    v4, _ = validate_file("tep_tin_khong_ton_tai_123.csv")
    if not v4:
        passed_cases += 1

    if passed_cases == 4:
        print("  ✓ Thành công: Cả 4 trường hợp ngoại lệ đều được phát hiện và ngăn chặn chính xác.")
        return True
    else:
        print(f"  ✗ Thất bại: Chỉ vượt qua {passed_cases}/4 trường hợp kiểm thử ngoại lệ.")
        return False


# ============================================================
# TODO 5 & 6: KIỂM TRA TOÀN BỘ HỆ THỐNG END-TO-END
# ============================================================

def test_end_to_end() -> bool:
    print("[TEST 5/5] Kiểm thử luồng tích hợp toàn hệ thống (End-to-End)...")
    try:
        # Giả lập luồng: UI gửi dữ liệu -> System gọi 3 mô hình -> Dashboard -> Báo cáo
        ecg_sig = np.random.randn(1000, 12).astype(np.float32)
        ppg_sig = 2000.0 + 200.0 * np.sin(2 * np.pi * 1.3 * np.linspace(0, 2.1, 2100))
        clin_data = {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
        }

        res_ecg = screen_ecg(ecg_sig)
        res_ppg = screen_ppg(ppg_sig)
        res_clin = screen_clinical(clin_data)

        # Tổng hợp dashboard
        dash = get_dashboard_data(ecg=res_ecg, ppg=res_ppg, clinical=res_clin)
        assert "overall_risk_score" in dash, "Thiếu 'overall_risk_score' trong dashboard"
        assert dash["total_modalities_available"] == 3, "Phải có đủ 3 nguồn dữ liệu khả dụng"

        # Xuất báo cáo y khoa
        report = create_full_report(
            ecg=res_ecg,
            ppg=res_ppg,
            clinical=res_clin,
            patient_info={"name": "Nguyễn Văn Test", "id": "TEST-001", "age": "63", "sex": "Nam"},
        )
        assert len(report) > 200, "Báo cáo sinh ra quá ngắn"
        assert "PHIẾU KẾT QUẢ" in report, "Tiêu đề báo cáo không chuẩn"

        print(f"  ✓ Thành công: Luồng tích hợp End-to-End hoàn chỉnh (Điểm rủi ro tổng thể: {dash['overall_risk_score']}/100 - {dash['risk_category']}).")
        return True
    except Exception as e:
        print(f"  ✗ Thất bại: {e}")
        traceback.print_exc()
        return False


def run_tests():
    """Chạy toàn bộ các ca kiểm thử của hệ thống."""
    print("=" * 65)
    print(" BẮT ĐẦU CHẠY KIỂM THỬ HỆ THỐNG (SYSTEM MONITORING)")
    print("=" * 65)
    results = [
        test_ecg(),
        test_ppg(),
        test_clinical(),
        test_invalid_input(),
        test_end_to_end(),
    ]
    passed = sum(results)
    total = len(results)
    print("=" * 65)
    print(f" KẾT QUẢ: ĐÃ VƯỢT QUA {passed}/{total} BỘ KIỂM THỬ ({passed/total*100:.0f}%)")
    print("=" * 65)
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
