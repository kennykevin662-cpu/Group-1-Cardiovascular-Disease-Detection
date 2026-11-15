"""THÀNH VIÊN 4 – UI – Clinical

Giao diện phân tích nguy cơ bệnh tim từ dữ liệu lâm sàng.
Tích hợp với System.screening.screen_clinical.
"""
import streamlit as st
import pandas as pd


def render_clinical_ui():
    st.markdown("## 📋 Phân tích Dữ liệu Lâm sàng (Clinical Screening)")
    st.markdown(
        "Nhập các chỉ số xét nghiệm, huyết động học và bệnh sử của bệnh nhân "
        "để mô hình AI đánh giá nguy cơ mắc bệnh tim mạch."
    )

    # Nút nạp nhanh dữ liệu mẫu
    col_demo1, col_demo2, _ = st.columns([1.5, 1.5, 4])
    with col_demo1:
        if st.button("📥 Nạp mẫu Nguy cơ thấp", use_container_width=True):
            st.session_state["demo_clin"] = {
                "age": 45, "sex": 0, "cp": 2, "trestbps": 120, "chol": 190,
                "fbs": 0, "restecg": 0, "thalach": 165, "exang": 0,
                "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 3
            }
            st.rerun()
    with col_demo2:
        if st.button("📥 Nạp mẫu Nguy cơ cao", use_container_width=True):
            st.session_state["demo_clin"] = {
                "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
                "fbs": 1, "restecg": 2, "thalach": 108, "exang": 1,
                "oldpeak": 2.5, "slope": 2, "ca": 3, "thal": 7
            }
            st.rerun()

    demo = st.session_state.get("demo_clin", {})

    with st.form("clinical_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 👤 Thông tin cơ bản & Huyết động")
            age = st.number_input("Tuổi (Age)", min_value=1, max_value=120, value=demo.get("age", 55), help="Tuổi của bệnh nhân (1 - 120)")
            sex_label = st.selectbox(
                "Giới tính (Sex)",
                ["Nam (1)", "Nữ (0)"],
                index=0 if demo.get("sex", 1) == 1 else 1,
            )
            sex = 1 if "Nam" in sex_label else 0

            cp_mapping = {
                "1: Đau thắt ngực điển hình (Typical Angina)": 1,
                "2: Đau thắt ngực không điển hình (Atypical Angina)": 2,
                "3: Đau không do tim (Non-anginal Pain)": 3,
                "4: Không có triệu chứng đau (Asymptomatic)": 4,
            }
            cp_idx = [1, 2, 3, 4].index(demo.get("cp", 4)) if demo.get("cp") in [1, 2, 3, 4] else 3
            cp_label = st.selectbox("Loại đau thắt ngực (Chest Pain Type)", list(cp_mapping.keys()), index=cp_idx)
            cp = cp_mapping[cp_label]

            trestbps = st.number_input(
                "Huyết áp lúc nghỉ (Resting Blood Pressure - mmHg)",
                min_value=50, max_value=260, value=demo.get("trestbps", 130),
                help="Huyết áp tâm thu đo lúc nghỉ ngơi."
            )
            chol = st.number_input(
                "Cholesterol huyết thanh (Serum Cholestoral - mg/dl)",
                min_value=50, max_value=600, value=demo.get("chol", 240),
                help="Mức mỡ máu Cholesterol."
            )
            fbs_label = st.radio(
                "Đường huyết lúc đói > 120 mg/dl (Fasting Blood Sugar)",
                ["Không (<= 120 mg/dl)", "Có (> 120 mg/dl)"],
                index=demo.get("fbs", 0),
                horizontal=True,
            )
            fbs = 1 if "Có" in fbs_label else 0

        with col2:
            st.markdown("#### ⚡ Điện tim & Gắng sức")
            restecg_map = {
                "0: Bình thường (Normal)": 0,
                "1: Sóng ST-T bất thường (ST-T wave abnormality)": 1,
                "2: Phì đại thất trái (Left ventricular hypertrophy)": 2,
            }
            recg_idx = [0, 1, 2].index(demo.get("restecg", 0)) if demo.get("restecg") in [0, 1, 2] else 0
            restecg_label = st.selectbox("Điện tâm đồ lúc nghỉ (Resting ECG)", list(restecg_map.keys()), index=recg_idx)
            restecg = restecg_map[restecg_label]

            thalach = st.number_input(
                "Nhịp tim tối đa đạt được khi gắng sức (Max Heart Rate - bpm)",
                min_value=40, max_value=240, value=demo.get("thalach", 150),
            )
            exang_label = st.radio(
                "Đau ngực khi gắng sức (Exercise Induced Angina)",
                ["Không (No)", "Có (Yes)"],
                index=demo.get("exang", 0),
                horizontal=True,
            )
            exang = 1 if "Có" in exang_label else 0

            oldpeak = st.number_input(
                "Đoạn ST chênh xuống khi gắng sức so với lúc nghỉ (ST Depression)",
                min_value=0.0, max_value=10.0, value=float(demo.get("oldpeak", 1.0)), step=0.1,
            )
            slope_map = {
                "1: Dốc lên (Upsloping)": 1,
                "2: Bằng phẳng (Flat)": 2,
                "3: Dốc xuống (Downsloping)": 3,
            }
            slope_idx = [1, 2, 3].index(demo.get("slope", 1)) if demo.get("slope") in [1, 2, 3] else 0
            slope_label = st.selectbox("Độ dốc đoạn ST đỉnh gắng sức (ST Slope)", list(slope_map.keys()), index=slope_idx)
            slope = slope_map[slope_label]

            ca = st.slider("Số lượng mạch máu lớn nhuộm màu huỳnh quang (Fluoroscopy Vessels - ca)", 0, 3, demo.get("ca", 0))

            thal_map = {
                "3: Bình thường (Normal)": 3,
                "6: Khiếm khuyết cố định (Fixed Defect)": 6,
                "7: Khiếm khuyết có thể hồi phục (Reversible Defect)": 7,
            }
            thal_keys = list(thal_map.keys())
            thal_idx = 0
            if demo.get("thal") == 6:
                thal_idx = 1
            elif demo.get("thal") == 7:
                thal_idx = 2
            thal_label = st.selectbox("Kết quả kiểm tra Thalassaemia (thal)", thal_keys, index=thal_idx)
            thal = thal_map[thal_label]

        submit = st.form_submit_button("🔍 Phân tích Nguy cơ Lâm sàng", use_container_width=True, type="primary")

    if submit:
        input_dict = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }

        with st.spinner("Đang xử lý chuẩn hóa và dự đoán..."):
            from System.screening import screen_clinical
            res = screen_clinical(input_dict)

        if res.get("status") == "success":
            st.session_state["clinical_result"] = res
            st.session_state["clinical_input"] = input_dict

            st.success("✅ Phân tích thành công!")
            prob = res["probability"]
            is_disease = (res["prediction"] == 1 or prob >= 0.5)

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Kết luận AI", res["label"])
            with col_res2:
                st.metric("Xác suất mắc bệnh tim", f"{prob:.1%}")
            with col_res3:
                st.metric("Mức độ tin cậy", f"{res['confidence']:.1%}")

            st.progress(prob)

            if is_disease:
                st.error(
                    f"⚠️ **Cảnh báo nguy cơ cao ({prob:.1%}):** Bệnh nhân có nhiều chỉ số phù hợp với bệnh lý mạch vành "
                    f"(Đau ngực, ST chênh: {oldpeak} mm, số mạch tổn thương: {ca}). Khuyến nghị thực hiện điện tâm đồ và chụp mạch vành."
                )
            else:
                st.info(
                    f"🟢 **Nguy cơ thấp ({prob:.1%}):** Các chỉ số sinh tồn và xét nghiệm hiện tại ở mức an toàn. "
                    "Khuyến nghị duy trì chế độ dinh dưỡng lành mạnh và theo dõi định kỳ."
                )
        else:
            st.error(f"❌ Lỗi: {res.get('message', res.get('detail', 'Không xác định'))}")


def main():
    render_clinical_ui()


if __name__ == "__main__":
    main()
