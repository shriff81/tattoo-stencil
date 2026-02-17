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
    
    st.sidebar.subheader("Тонкая настройка линий")
    edge_sensitivity = st.sidebar.slider("Чувствительность к деталям", 10, 250, 150)
    shadow_depth = st.sidebar.slider("Глубина теней", 1, 15, 5)

    # 1. Обработка: Улучшенное выделение границ
    # Используем Bilateral Filter для сохранения границ при удалении шума
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Основные контуры
    edges = cv2.Canny(smooth, edge_sensitivity // 2, edge_sensitivity)
    
    # Теневые переходы (изолинии)
    shadow_map = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, shadow_depth)
    shadow_edges = cv2.Canny(shadow_map, 10, 50)
    
    # Объединяем все линии
    combined_edges = cv2.bitwise_or(edges, shadow_edges)

    # 2. Создание Красного Стенсила
    # Создаем пустое изображение с Альфа-каналом (RGBA)
    h, w = gray.shape
    red_stencil = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Заполняем линии ярко-красным (R:255, G:0, B:0)
    red_stencil[combined_edges > 0] = [255, 0, 0, 255] 
    
    if bg_mode == "Белый (для печати)":
        # Создаем белый фон и накладываем на него красные линии
        final_view = np.ones((h, w, 3), dtype=np.uint8) * 255
        final_view[combined_edges > 0] = [255, 0, 0]
    else:
        # Оставляем прозрачным там, где нет линий
        final_view = red_stencil

    st.image(final_view, caption="Красный стенсил (Превью)", use_column_width=True)

    # Функция PDF
    def create_pdf(img_np, width_cm, is_transparent):
        h, w = img_np.shape[:2]
        aspect = h / w
        height_cm = width_cm * aspect
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        
        # Если был прозрачный, для PDF делаем на белом (принтеры не печатают прозрачность)
        if is_transparent:
            pdf_img_np = np.ones((h, w, 3), dtype=np.uint8) * 255
            pdf_img_np[combined_edges > 0] = [255, 0, 0]
        else:
            pdf_img_np = img_np

        temp_img = Image.fromarray(pdf_img_np)
        img_byte_arr = io.BytesIO()
        temp_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        from reportlab.lib.utils import ImageReader
        p.drawImage(ImageReader(img_byte_arr), 1*cm, (29.7 - height_cm - 1)*cm, width=width_cm*cm, height=height_cm*cm)
        p.showPage()
        p.save()
        return buffer.getvalue()

    pdf_data = create_pdf(final_view, target_width_cm, bg_mode == "Прозрачный")
    
    st.download_button(
        label="📥 Скачать КРАСНЫЙ PDF (1:1)",
        data=pdf_data,
        file_name="angar_red_stencil.pdf",
        mime="application/pdf"
    )
