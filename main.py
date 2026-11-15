"""Điểm khởi chạy dự án - Thành viên 4: System & UI.

Hướng dẫn khởi chạy:
  Cách 1 (Khuyên dùng): streamlit run UI/app.py
  Cách 2: python main.py --run
  Cách 3 (Chạy kiểm thử hệ thống): python main.py --test
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Đảm bảo bảng mã UTF-8 trên console Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_system_health():
    """Kiểm tra các mô hình và tài nguyên cần thiết."""
    print("=" * 65)
    print(" KIỂM TRA TRẠNG THÁI HỆ THỐNG (SYSTEM HEALTH CHECK)")
    print("=" * 65)

    items = [
        ("Mô hình Clinical", ROOT / "mohinh" / "Clinical" / "clinical_model.pkl"),
        ("Bộ chuẩn hóa Clinical", ROOT / "mohinh" / "Clinical" / "clinical_scaler.pkl"),
        ("Mô hình ECG", ROOT / "mohinh" / "ECG" / "ecg_model.pkl"),
        ("Mô hình PPG", ROOT / "mohinh" / "PPG" / "ppg_model.pkl"),
        ("Giao diện UI chính", ROOT / "UI" / "app.py"),
    ]

    all_ready = True
    for name, path in items:
        exists = path.exists()
        status = "✅ Sẵn sàng" if exists else "❌ Thiếu tệp"
        print(f"- {name:<25}: {status} ({path.relative_to(ROOT)})")
        if not exists:
            all_ready = False

    print("=" * 65)
    return all_ready


def main():
    parser = argparse.ArgumentParser(description="Khởi chạy Hệ thống AI Đa phương thức Nhóm 1")
    parser.add_argument("--run", action="store_true", help="Tự động khởi chạy giao diện Streamlit")
    parser.add_argument("--test", action="store_true", help="Chạy toàn bộ 5 bộ kiểm thử hệ thống")
    args = parser.parse_args()

    ready = check_system_health()

    if args.test:
        from System.monitoring import run_tests
        run_tests()
        return

    if args.run:
        app_path = ROOT / "UI" / "app.py"
        print(f"\n🚀 Đang khởi chạy giao diện Streamlit tại: {app_path} ...")
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
        subprocess.run(cmd)
        return

    print("\n💡 HƯỚNG DẪN KHỞI CHẠY HỆ THỐNG:")
    print("1. Khởi chạy giao diện người dùng web (Streamlit):")
    print("     streamlit run UI/app.py")
    print("   hoặc:")
    print("     python main.py --run")
    print("\n2. Chạy bộ kiểm thử tự động toàn diện:")
    print("     python main.py --test\n")


if __name__ == "__main__":
    main()
