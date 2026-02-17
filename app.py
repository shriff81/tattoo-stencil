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
    
    st.sidebar.header("Настройки Мастера")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    bg_mode = st.sidebar.radio("Фон стенсила:", ["Белый (для печати)", "Прозрачный"])
    
    st.sidebar.subheader("Детализация")
    # Увеличил диапазон, чтобы вы могли "вытащить" даже самые мелкие детали
    sensitivity = st.sidebar.slider("Чувствительность к мелким деталям", 3, 25, 9, step=2)

    # 1. Подготовка фото (убираем шум, сохраняя резкость)
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    
    # 2. Адаптивный метод (ищет линии, а не двойные границы)
    # Это позволяет избежать "обводки контура"
    thresh = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, sensitivity, 2)
    
    # Инвертируем, чтобы получить линии
    lines = cv2.bitwise_not(thresh)
    
    # Небольшая очистка от "мусора" (одиночных пикселей)
    kernel = np.ones((2,2), np.uint8)
    final_edges = cv2.morphologyEx(lines, cv2.MORPH_OPEN, kernel)

    # 3. Создание финального изображения
    h, w = gray.shape
    if bg_mode == "Белый (для печати)":
        final_view = np.ones((h, w, 3), dtype=np.uint8) * 255
        final_view[final_edges > 0] = [255, 0, 0] # Ярко-красный
    else:
        final_view = np.zeros((h, w, 4), dtype=np.uint8)
        final_view[final_edges > 0] = [255, 0, 0, 255]

    st.image(final_view, caption="Красный стенсил", use_column_width=True)

    # PDF Функция
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
        # Центрируем на листе
        p.drawImage(ImageReader(img_byte_arr), 1*cm, (29.7 - height_cm - 1)*cm, width=width_cm*cm, height=height_cm*cm)
        p.showPage()
        p.save()
        return buffer.getvalue()

    pdf_data = create_pdf(final_edges, target_width_cm)
    
    st.download_button(
        label="📥 Скачать КРАСНЫЙ PDF (1:1)",
        data=pdf_data,
        file_name="angar_red_stencil.pdf",
        mime="application/pdf"
    )
