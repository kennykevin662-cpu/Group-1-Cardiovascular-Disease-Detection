"""THÀNH VIÊN 4 – UI
Ứng dụng chính điều hướng và tích hợp toàn bộ hệ sinh thái AI phát hiện bệnh tim mạch:
- Clinical (Dữ liệu lâm sàng)
- ECG (Điện tâm đồ 12 chuyển đạo)
- PPG (Quang thể tích & nhịp mạch)
- Dashboard tổng hợp & Báo cáo chẩn đoán
- Giám sát & Kiểm thử hệ thống
"""
import sys
from pathlib import Path

import streamlit as st

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Cardiovascular AI System - Group 1",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from System.dashboard import get_dashboard_data
from System.monitoring import run_tests
from System.report import create_full_report
from UI.clinical_app import render_clinical_ui
from UI.ecg_app import render_ecg_ui
from UI.ppg_app import render_ppg_ui


def init_session():
    """Khởi tạo trạng thái phiên làm việc."""
    if "patient_name" not in st.session_state:
        st.session_state["patient_name"] = "Nguyễn Văn An"
    if "patient_id" not in st.session_state:
        st.session_state["patient_id"] = "BN-202611"
    if "patient_age" not in st.session_state:
        st.session_state["patient_age"] = "58"
    if "patient_sex" not in st.session_state:
        st.session_state["patient_sex"] = "Nam"


def render_dashboard():
    """Hiển thị bảng điều khiển tổng hợp đa phương thức."""
    st.markdown("## 📊 Bảng điều khiển Đa phương thức & Báo cáo Tổng hợp")
    st.markdown(
        "Hệ thống tổng hợp kết quả phân tích từ **3 nguồn dữ liệu độc lập** "
        "(Điện tâm đồ ECG, Quang thể tích PPG và Chỉ số Lâm sàng) để đánh giá "
        "nguy cơ tim mạch toàn diện cho bệnh nhân."
    )

    ecg_res = st.session_state.get("ecg_result")
    ppg_res = st.session_state.get("ppg_result")
    clin_res = st.session_state.get("clinical_result")

    dash = get_dashboard_data(ecg=ecg_res, ppg=ppg_res, clinical=clin_res)

    # 1. Thẻ tổng quan chỉ số rủi ro
    st.markdown("---")
    score_col, status_col, modal_col = st.columns([2, 3, 2])

    with score_col:
        st.metric(
            label="Chỉ số Rủi ro Tim mạch Toàn diện (CRI)",
            value=f"{dash['overall_risk_score']}/100",
            delta=f"{dash['overall_risk_score'] - 50.0:.1f} điểm chuẩn",
            delta_color="inverse",
        )
        st.progress(dash["overall_risk_score"] / 100.0)

    with status_col:
        st.markdown(f"#### Phân loại: **{dash['risk_category']}**")
        st.info(f"💡 **Khuyến nghị**: {dash['advice']}")

    with modal_col:
        avail = dash["total_modalities_available"]
        abnorm = dash["abnormal_modalities_count"]
        st.metric("Dữ liệu đã phân tích", f"{avail}/3 phân hệ")
        st.metric("Phân hệ ghi nhận bất thường", f"{abnorm}/{avail}" if avail > 0 else "0/0")

    # 2. Chi tiết trạng thái từng phân hệ
    st.markdown("---")
    st.markdown("### 🔍 Trạng thái chi tiết từng phân hệ")
    col_c, col_e, col_p = st.columns(3)

    with col_c:
        st.markdown("#### 📋 Dữ liệu Lâm sàng (UCI)")
        c_info = dash["clinical"]
        if c_info["available"]:
            st.success(f"**Kết quả**: {c_info['label']}")
            st.write(f"- Xác suất có bệnh: **{c_info['probability']:.1%}**")
            st.write(f"- Đánh giá: {'🔴 Nguy cơ bệnh mạch vành' if c_info['is_diseased'] else '🟢 An toàn'}")
        else:
            st.warning("Chưa có dữ liệu lâm sàng. Vui lòng vào mục *'📋 Dữ liệu Lâm sàng'* để nhập.")

    with col_e:
        st.markdown("#### 🫀 Điện tâm đồ (PTB-XL)")
        e_info = dash["ecg"]
        if e_info["available"]:
            st.success(f"**Kết quả**: {e_info['label']}")
            st.write(f"- Xác suất bất thường: **{e_info['probability']:.1%}**")
            hr_txt = f"{e_info['heart_rate']} bpm" if e_info.get("heart_rate") else "N/A"
            st.write(f"- Nhịp tim Lead II: **{hr_txt}**")
            st.write(f"- Đánh giá: {'🔴 Phát hiện bất thường sóng ECG' if e_info['is_abnormal'] else '🟢 Điện tim bình thường'}")
        else:
            st.warning("Chưa có dữ liệu ECG. Vui lòng vào mục *'🫀 Sàng lọc ECG'* để tải sóng.")

    with col_p:
        st.markdown("#### 📈 Quang thể tích (PPG-BP)")
        p_info = dash["ppg"]
        if p_info["available"]:
            st.success(f"**Trạng thái**: {p_info['label']}")
            st.write(f"- Nhịp tim AI ước lượng: **{p_info['heart_rate']} bpm**")
            hr_p_txt = f"{p_info['heart_rate_peaks']} bpm" if p_info.get("heart_rate_peaks") else "N/A"
            st.write(f"- Nhịp tim đo từ đỉnh xung: **{hr_p_txt}**")
            st.write(f"- Đánh giá: {'🔴 Rối loạn nhịp mạch' if p_info['is_abnormal'] else '🟢 Nhịp đập ổn định'}")
        else:
            st.warning("Chưa có dữ liệu PPG. Vui lòng vào mục *'📈 Sàng lọc PPG'* để tải sóng.")

    # 3. Xuất và tải báo cáo y khoa
    st.markdown("---")
    st.markdown("### 📄 Xuất Báo cáo Y khoa Toàn diện")
    patient_info = {
        "name": st.session_state.get("patient_name", "N/A"),
        "id": st.session_state.get("patient_id", "N/A"),
        "age": st.session_state.get("patient_age", "N/A"),
        "sex": st.session_state.get("patient_sex", "N/A"),
    }

    full_report_text = create_full_report(
        ecg=ecg_res,
        ppg=ppg_res,
        clinical=clin_res,
        patient_info=patient_info,
    )

    with st.expander("👁️ Xem trước Phiếu Báo cáo Y tế (Markdown Preview)"):
        st.markdown(full_report_text)

    col_btn1, col_btn2, _ = st.columns([2, 2, 3])
    with col_btn1:
        st.download_button(
            label="📥 Tải Phiếu Báo cáo (.md)",
            data=full_report_text,
            file_name=f"Bao_cao_tim_mach_{patient_info['id']}.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary",
        )


def render_monitoring():
    """Hiển thị trang giám sát và chạy kiểm thử tự động."""
    st.markdown("## ⚙️ Giám sát & Kiểm thử Hệ thống (System Monitoring)")
    st.markdown(
        "Mô-đun kiểm thử tự động (Unit Test & Integration Test) kiểm tra sự sẵn sàng "
        "của cả 3 mô hình học máy, các hàm tiền xử lý dữ liệu và luồng liên kết end-to-end."
    )

    if st.button("▶️ Chạy toàn bộ 5 bộ kiểm thử hệ thống ngay", type="primary"):
        with st.spinner("Đang chạy kiểm thử tự động..."):
            from System.monitoring import test_clinical, test_ecg, test_end_to_end, test_invalid_input, test_ppg

            t1 = test_ecg()
            t2 = test_ppg()
            t3 = test_clinical()
            t4 = test_invalid_input()
            t5 = test_end_to_end()

        col1, col2 = st.columns(2)
        with col1:
            st.write("1. Mô hình ECG (ecg_model.pkl & 89 đặc trưng):", "✅ ĐẠT" if t1 else "❌ THẤT BẠI")
            st.write("2. Mô hình PPG (ppg_model.pkl & nhịp tim):", "✅ ĐẠT" if t2 else "❌ THẤT BẠI")
            st.write("3. Mô hình Clinical (clinical_model.pkl):", "✅ ĐẠT" if t3 else "❌ THẤT BẠI")
        with col2:
            st.write("4. Kiểm tra bắt lỗi đầu vào ngoại lệ:", "✅ ĐẠT" if t4 else "❌ THẤT BẠI")
            st.write("5. Tích hợp toàn hệ thống End-to-End:", "✅ ĐẠT" if t5 else "❌ THẤT BẠI")

        if all([t1, t2, t3, t4, t5]):
            st.success("🎉 TẤT CẢ 5/5 BỘ KIỂM THỬ ĐÃ VƯỢT QUA 100%! HỆ THỐNG ĐANG Ở TRẠNG THÁI TỐT NHẤT.")
        else:
            st.error("Có bộ kiểm thử chưa vượt qua. Vui lòng kiểm tra lại log chi tiết.")


def main():
    init_session()

    # Thanh điều hướng bên trái (Sidebar)
    with st.sidebar:
        st.markdown("# 🫀 CardioAI System")
        st.caption("Nhóm 1 - Phát hiện Bệnh Tim Mạch Đa Phương Thức")
        st.markdown("---")

        st.markdown("### 👤 Hồ sơ Bệnh nhân")
        st.session_state["patient_name"] = st.text_input("Họ và tên:", st.session_state["patient_name"])
        col_id, col_age = st.columns(2)
        with col_id:
            st.session_state["patient_id"] = st.text_input("Mã BN:", st.session_state["patient_id"])
        with col_age:
            st.session_state["patient_age"] = st.text_input("Tuổi:", st.session_state["patient_age"])
        st.session_state["patient_sex"] = st.selectbox(
            "Giới tính:", ["Nam", "Nữ"],
            index=0 if st.session_state["patient_sex"] == "Nam" else 1,
        )

        st.markdown("---")
        st.markdown("### 🧭 Điều hướng chức năng")
        menu = st.radio(
            "Chọn phân hệ làm việc:",
            [
                "📊 Bảng điều khiển & Báo cáo",
                "📋 Dữ liệu Lâm sàng (Clinical)",
                "🫀 Sàng lọc Điện tâm đồ (ECG)",
                "📈 Sàng lọc Quang thể tích (PPG)",
                "⚙️ Kiểm thử & Giám sát (Monitoring)",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("**Thành viên 4**: System & UI  \n*Streamlit 1.67.0 | Python 3.14*")

    # Hiển thị nội dung theo trang đã chọn
    if "Bảng điều khiển" in menu:
        render_dashboard()
    elif "Lâm sàng" in menu:
        render_clinical_ui()
    elif "Điện tâm đồ" in menu:
        render_ecg_ui()
    elif "Quang thể tích" in menu:
        render_ppg_ui()
    elif "Kiểm thử" in menu:
        render_monitoring()


if __name__ == "__main__":
    main()
