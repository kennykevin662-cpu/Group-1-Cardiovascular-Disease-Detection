"""THÀNH VIÊN 4 – UI – PPG

Giao diện phân tích tín hiệu quang thể tích (PPG).
Tích hợp với System.screening.screen_ppg.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def render_ppg_ui():
    st.markdown("## 📈 Sàng lọc Quang thể tích (PPG Screening)")
    st.markdown(
        "Tải lên bản ghi sóng xung quang thể tích PPG (từ cảm biến quang học ngón tay/đồng hồ thông minh) "
        "để mô hình AI ước lượng nhịp tim, phát hiện đỉnh xung tâm thu và phân tích biến thiên nhịp mạch."
    )

    tab_sample, tab_upload = st.tabs(["📁 Chọn mẫu đo PPG-BP có sẵn", "📤 Tải file PPG lên (.txt / .csv / .npy)"])
    fs = 1000  # Chuẩn PPG-BP 1000 Hz

    with tab_sample:
        ppg_dir = ROOT / "dataset" / "PPG" / "Data File" / "0_subject"
        sample_files = sorted(list(ppg_dir.glob("*.txt")))[:20] if ppg_dir.exists() else []

        if sample_files:
            file_names = [f.name for f in sample_files]
            chosen_file = st.selectbox("Chọn tệp mẫu PPG-BP thực tế:", file_names)

            if st.button("📥 Nạp dữ liệu mẫu PPG này", key="btn_load_sample_ppg"):
                target_path = ppg_dir / chosen_file
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    vals = [float(x) for x in f.read().split() if x]
                st.session_state["current_ppg_signal"] = np.array(vals, dtype=np.float32)
                st.session_state["current_ppg_source"] = chosen_file
                st.success(f"Đã nạp tệp: {chosen_file} ({len(vals)} điểm đo).")
        else:
            st.info("Đang tạo mẫu mô phỏng do không tìm thấy thư mục tệp PPG thô.")
            if st.button("Tạo sóng PPG mô phỏng", key="btn_sim_ppg"):
                t = np.linspace(0, 2.1, 2100)
                sim = 2000.0 + 350.0 * np.sin(2 * np.pi * 1.25 * t) + np.random.randn(2100) * 15
                st.session_state["current_ppg_signal"] = sim.astype(np.float32)
                st.session_state["current_ppg_source"] = "Sóng PPG mô phỏng (Simulated)"
                st.success("Đã nạp mẫu sóng mô phỏng.")

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Tải lên tệp tín hiệu PPG:",
            type=["txt", "csv", "npy"],
            help="Tệp văn bản chứa chuỗi số mẫu sóng hoặc file .csv/.npy"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".npy"):
                    loaded_sig = np.load(uploaded_file).reshape(-1)
                elif uploaded_file.name.endswith(".txt"):
                    content = uploaded_file.read().decode("utf-8", errors="ignore")
                    loaded_sig = np.array([float(x) for x in content.split() if x], dtype=np.float32)
                else:
                    import pandas as pd
                    df = pd.read_csv(uploaded_file)
                    col = [c for c in df.columns if any(k in str(c).lower() for k in ("ppg", "pleth"))]
                    chosen_col = col[0] if col else df.columns[0]
                    loaded_sig = pd.to_numeric(df[chosen_col], errors="coerce").dropna().to_numpy()

                st.session_state["current_ppg_signal"] = loaded_sig
                st.session_state["current_ppg_source"] = uploaded_file.name
                st.success(f"Đã tải tệp: {uploaded_file.name} ({len(loaded_sig)} mẫu).")
            except Exception as e:
                st.error(f"Lỗi đọc tệp PPG: {e}")

    active_sig = st.session_state.get("current_ppg_signal")
    active_source = st.session_state.get("current_ppg_source", "Chưa xác định")

    if active_sig is not None:
        st.markdown("---")
        st.markdown(f"### 📊 Phân tích dạng sóng PPG: `{active_source}`")

        # Vẽ biểu đồ sóng PPG và phát hiện đỉnh tâm thu
        clean_sig = (active_sig - np.mean(active_sig)) / (np.std(active_sig) + 1e-8)
        peaks, _ = find_peaks(clean_sig, distance=max(1, int(0.3 * fs)), prominence=0.3)

        fig, ax = plt.subplots(figsize=(11, 3.2))
        t_axis = np.arange(len(active_sig)) / float(fs)

        ax.plot(t_axis, clean_sig, color="#1d3557", linewidth=1.2, label="Sóng PPG chuẩn hóa")
        if len(peaks) > 0:
            ax.scatter(t_axis[peaks], clean_sig[peaks], color="#e63946", s=60, zorder=5, label=f"Đỉnh tâm thu ({len(peaks)} đỉnh)")

        ax.set_title(f"Dạng sóng xung quang học PPG - Tần số {fs} Hz", fontsize=11, fontweight="bold")
        ax.set_xlabel("Thời gian (giây)")
        ax.set_ylabel("Biên độ chuẩn hóa")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Nút thực hiện phân tích
        if st.button("🚀 Bắt đầu Phân tích Nhịp tim & PPG bằng AI", type="primary", use_container_width=True):
            with st.spinner("Đang lọc thông dải 0.5-8Hz, trích xuất đặc trưng và dự đoán qua Random Forest Regressor..."):
                from System.screening import screen_ppg
                res = screen_ppg(active_sig, fs=fs)

            if res.get("status") == "success":
                st.session_state["ppg_result"] = res
                st.success("✅ Phân tích PPG hoàn tất!")

                hr_pred = res["heart_rate"]
                hr_peaks = res["heart_rate_peaks"]
                status_label = res["label"]
                risk_lvl = res["risk_level"]

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nhịp tim AI ước lượng", f"{hr_pred} bpm")
                with col2:
                    st.metric("Nhịp tim đo từ đỉnh xung", f"{hr_peaks} bpm")
                with col3:
                    st.metric("Trạng thái nhịp tim", status_label)
                with col4:
                    rmssd_disp = f"{res['rmssd']:.4f} s" if res.get("rmssd") else "N/A"
                    st.metric("Chỉ số HRV (RMSSD)", rmssd_disp)

                if risk_lvl != "Bình thường":
                    st.warning(
                        f"⚠️ **Cảnh báo nhịp tim ({hr_pred} bpm):** {status_label}. "
                        "Nhịp tim nghỉ ngơi vượt ra ngoài giới hạn bình thường (60 - 100 bpm). "
                        "Cần nghỉ ngơi, đo lại và tham vấn bác sĩ nếu tình trạng kéo dài kèm chóng mặt, khó thở."
                    )
                else:
                    st.info(
                        f"🟢 **Nhịp tim bình thường ({hr_pred} bpm):** Tần số tim ổn định, "
                        f"biên độ xung nhịp đều đặn ({res.get('details', {}).get('peak_count', len(peaks))} nhịp trong cửa sổ đo)."
                    )
            else:
                st.error(f"Lỗi phân tích: {res.get('message', res.get('detail', 'Không xác định'))}")


def main():
    render_ppg_ui()


if __name__ == "__main__":
    main()
