import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

st.set_page_config(page_title="AnGar Stencil Pro", layout="wide")
st.title("AnGar Stencil Pro 🔴")

uploaded_file = st.file_uploader("Загрузите референс", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Настройки в боковой панели
    st.sidebar.header("Настройки Мастера")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    bg_mode = st.sidebar.radio("Фон стенсила:", ["Белый (для печати)", "Прозрачный"])
    
    st.sidebar.subheader("Тонкая настройка")
    # Возвращаем ползунок чувствительности для тонких линий
    edge_sensitivity = st.sidebar.slider("Чувствительность контуров", 10, 250, 120)

    # 1. Обработка: Тонкие линии через Canny
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(smooth, edge_sensitivity // 2, edge_sensitivity)

    # 2. Создание Красного Стенсила
    h, w = gray.shape
    if bg_mode == "Белый (для печати)":
        final_view = np.ones((h, w, 3), dtype=np.uint8) * 255
        final_view[edges > 0] = [255, 0, 0] # Ярко-красный
    else:
        # RGBA для прозрачности
        final_view = np.zeros((h, w, 4), dtype=np.uint8)
        final_view[edges > 0] = [255, 0, 0, 255]

    st.image(final_view, caption="Красный стенсил (Тонкие линии)", use_column_width=True)

    # Функция PDF
    def create_pdf(edge_data, width_cm):
        h, w = edge_data.shape
        aspect = h / w
        height_cm = width_cm * aspect
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        
        pdf_img_np = np.ones((h, w, 3), dtype=np.uint8) * 255
        pdf_img_np[edge_data > 0] = [255, 0, 0]
        
        temp_img = Image.fromarray(pdf_img_np)
        img_byte_arr = io.BytesIO()
        temp_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        from reportlab.lib.utils import ImageReader
        p.drawImage(ImageReader(img_byte_arr), 1*cm, (29.7 - height_cm - 1)*cm, width=width_cm*cm, height=height_cm*cm)
        p.showPage()
        p.save()
        return buffer.getvalue()

    pdf_data = create_pdf(edges, target_width_cm)
    
    st.download_button(
        label="📥 Скачать КРАСНЫЙ PDF (1:1)",
        data=pdf_data,
        file_name="angar_red_stencil.pdf",
        mime="application/pdf"
    )
