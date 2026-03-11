import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Cấu hình tiêu đề Web
st.set_page_config(page_title="Hệ thống Nhận diện AI", page_icon="🤖")
st.title("🚀 Web App Nhận Diện Đồ Vật bằng AI")
st.write("Dự án ứng dụng Computer Vision. Vui lòng tải ảnh lên để AI phân tích!")

# Tải model (Đã dùng mẹo đường dẫn tuyệt đối để chống lỗi)
@st.cache_resource
def load_model():
    # Lấy địa chỉ thư mục hiện tại ghép với tên file pt
    file_path = os.path.join(os.path.dirname(__file__), 'best-Beverage Containers-yolov8.pt')
    return YOLO(file_path)

model = load_model()

# Tạo nút Upload
uploaded_file = st.file_uploader("Tải lên một bức ảnh (jpg, png)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    if st.button("Bắt đầu Nhận diện (Run Inference)"):
        with st.spinner('AI đang xử lý...'):
            img_array = np.array(image)
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Chạy AI
            results = model(img_cv2, conf=0.5)
            annotated_frame = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            st.image(annotated_rgb, caption="Kết quả từ AI", use_container_width=True)
            st.success("✅ Hoàn tất!")
