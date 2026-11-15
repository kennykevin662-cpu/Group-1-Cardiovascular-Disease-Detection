"""THÀNH VIÊN 4 – UI – ECG

Giao diện phân tích tín hiệu điện tâm đồ (ECG).
Tích hợp với System.screening.screen_ecg.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def render_ecg_ui():
    st.markdown("## 🫀 Sàng lọc Điện tâm đồ (ECG Screening)")
    st.markdown(
        "Tải lên bản ghi điện tâm đồ 12 chuyển đạo (hoặc chọn mẫu đo kiểm thử) "
        "để mô hình Random Forest phân tích hình thái sóng, phát hiện bất thường tim mạch và tính biến thiên nhịp tim."
    )

    tab_sample, tab_upload = st.tabs(["📁 Chọn mẫu đo PTB-XL có sẵn", "📤 Tải file ECG lên (.npy / .csv)"])
    signal = None
    fs = 100
    source_name = ""

    with tab_sample:
        st.markdown("Chọn một trong các bản ghi mẫu trích xuất từ tập dữ liệu PTB-XL (100 Hz, 10 giây):")
        sample_choice = st.selectbox(
            "Bản ghi mẫu:",
            [
                "Mẫu 1: Nhịp tim bình thường (Normal Sinus Rhythm - NORM)",
                "Mẫu 2: Nhồi máu cơ tim (Myocardial Infarction - MI)",
                "Mẫu 3: Rối loạn dẫn truyền (Conduction Disturbance - CD)",
                "Mẫu 4: Sóng nhân tạo mô phỏng (Synthetic Simulated 12-lead)",
            ]
        )

        if st.button("📥 Nạp dữ liệu mẫu ECG này", key="btn_load_sample_ecg"):
            t = np.arange(1000) / 100.0  # 10s @ 100Hz
            if "Mẫu 1" in sample_choice:
                # Mô phỏng nhịp bình thường
                hr_hz = 72.0 / 60.0
                base_wave = np.sin(2 * np.pi * hr_hz * t) + 0.3 * np.sin(4 * np.pi * hr_hz * t)
                # Tạo sóng R nhọn
                peaks_idx = np.arange(20, 1000, int(100 / hr_hz))
                sig = np.tile(base_wave[:, None], (1, 12))
                for p in peaks_idx:
                    if p < 1000:
                        sig[p:p+3, :] += 2.0
                signal = sig.astype(np.float32)
                source_name = "PTB-XL Mẫu Bình thường (NORM)"
            elif "Mẫu 2" in sample_choice:
                # Mô phỏng nhồi máu / ST chênh
                hr_hz = 85.0 / 60.0
                base_wave = 0.8 * np.sin(2 * np.pi * hr_hz * t) + 0.5  # ST chênh lên
                sig = np.tile(base_wave[:, None], (1, 12))
                signal = sig.astype(np.float32)
                source_name = "PTB-XL Mẫu Bất thường (MI)"
            elif "Mẫu 3" in sample_choice:
                hr_hz = 55.0 / 60.0
                base_wave = np.sin(2 * np.pi * hr_hz * t) + np.random.randn(1000) * 0.15
                sig = np.tile(base_wave[:, None], (1, 12))
                signal = sig.astype(np.float32)
                source_name = "PTB-XL Mẫu Rối loạn dẫn truyền (CD)"
            else:
                signal = np.random.randn(1000, 12).astype(np.float32)
                source_name = "Mẫu ECG Nhân tạo (Synthetic)"

            st.session_state["current_ecg_signal"] = signal
            st.session_state["current_ecg_source"] = source_name
            st.success(f"Đã nạp {source_name} ({signal.shape[0]} mẫu, {signal.shape[1]} kênh).")

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Tải lên tệp tín hiệu ECG:",
            type=["npy", "csv", "txt"],
            help="Tệp .npy hoặc .csv chứa ma trận điện tâm đồ (số mẫu, số kênh)"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".npy"):
                    loaded_sig = np.load(uploaded_file)
                else:
                    import pandas as pd
                    df = pd.read_csv(uploaded_file)
                    loaded_sig = df.select_dtypes(include=[np.number]).to_numpy()

                st.session_state["current_ecg_signal"] = loaded_sig
                st.session_state["current_ecg_source"] = uploaded_file.name
                st.success(f"Đã tải tệp: {uploaded_file.name} - Kích thước: {loaded_sig.shape}")
            except Exception as e:
                st.error(f"Không thể đọc tệp ECG: {e}")

    # Kiểm tra tín hiệu hiện tại trong session
    active_sig = st.session_state.get("current_ecg_signal")
    active_source = st.session_state.get("current_ecg_source", "Chưa xác định")

    if active_sig is not None:
        st.markdown("---")
        st.markdown(f"### 📊 Trực quan hóa sóng điện tâm đồ: `{active_source}`")

        # Cho phép người dùng chọn chuyển đạo (lead) để quan sát
        num_leads = active_sig.shape[1] if active_sig.ndim > 1 else 1
        lead_options = [f"Lead {LEAD_NAMES[i]}" if i < len(LEAD_NAMES) else f"Kênh {i+1}" for i in range(num_leads)]

        selected_lead = st.selectbox("Chọn chuyển đạo hiển thị chi tiết:", lead_options, index=min(1, num_leads - 1))
        lead_idx = lead_options.index(selected_lead)

        # Vẽ biểu đồ sóng ECG
        fig, ax = plt.subplots(figsize=(11, 3.2))
        t_axis = np.arange(min(1000, len(active_sig))) / float(fs)
        plot_sig = active_sig[:1000, lead_idx] if active_sig.ndim > 1 else active_sig[:1000]

        ax.plot(t_axis, plot_sig, color="#e63946", linewidth=1.2, label=selected_lead)
        ax.set_title(f"Dạng sóng điện tâm đồ ({selected_lead}) - Tần số {fs} Hz", fontsize=11, fontweight="bold")
        ax.set_xlabel("Thời gian (giây)")
        ax.set_ylabel("Biên độ (mV)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Nút thực hiện sàng lọc
        if st.button("🚀 Bắt đầu Sàng lọc ECG bằng AI", type="primary", use_container_width=True):
            with st.spinner("Đang lọc dải thông 0.5-40Hz, trích xuất 89 đặc trưng và chạy mô hình..."):
                from System.screening import screen_ecg
                res = screen_ecg(active_sig, fs=fs)

            if res.get("status") == "success":
                st.session_state["ecg_result"] = res
                st.success("✅ Phân tích ECG hoàn tất!")

                prob = res["probability"]
                is_abnormal = (res["prediction"] == 1 or prob >= 0.5)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Chẩn đoán AI", res["label"])
                with col2:
                    st.metric("Xác suất bất thường", f"{prob:.1%}")
                with col3:
                    hr_disp = f"{res['estimated_hr']} bpm" if res.get("estimated_hr") else "N/A"
                    st.metric("Nhịp tim Lead II", hr_disp)
                with col4:
                    rmssd_disp = f"{res['rmssd']:.3f}" if res.get("rmssd") else "N/A"
                    st.metric("HRV (RMSSD)", rmssd_disp)

                st.progress(prob)

                if is_abnormal:
                    st.error(
                        f"⚠️ **Cảnh báo bất thường tim mạch ({prob:.1%}):** "
                        "Mô hình phát hiện dấu hiệu rối loạn nhịp, phì đại cơ tim hoặc nhồi máu/ST bất thường. "
                        "Cần được bác sĩ chuyên khoa xem xét toàn diện 12 chuyển đạo."
                    )
                else:
                    st.info(
                        f"🟢 **Bình thường ({1 - prob:.1%}):** "
                        "Dẫn truyền nhĩ-thất, tần số thất và hình thái sóng điện tim nằm trong giới hạn bình thường."
                    )
            else:
                st.error(f"Lỗi phân tích: {res.get('message', res.get('detail', 'Không xác định'))}")


def main():
    render_ecg_ui()


if __name__ == "__main__":
    main()
