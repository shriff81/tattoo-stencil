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
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    st.sidebar.header("Настройки Мастера")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    
    stencil_color_name = st.sidebar.selectbox(
        "Цвет стенсила:",
        ["Ярко-красный", "Ярко-синий", "Ярко-зеленый", "Черный"]
    )
    
    colors_dict = {
        "Ярко-красный": [255, 0, 0], "Ярко-синий": [0, 0, 255],
        "Ярко-зеленый": [0, 255, 0], "Черный": [0, 0, 0]
    }
    selected_color = colors_dict[stencil_color_name]

    st.sidebar.subheader("Визуальный контроль")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)
    
    st.sidebar.subheader("Геометрия линий")
    shadow_detail = st.sidebar.slider("Проработка теней (детализация)", 0, 10, 2)
    noise_reduction = st.sidebar.slider("Чистота фона", 1, 10, 2)
    edge_sensitivity = st.sidebar.slider("Чувствительность", 10, 250, 140)
    line_thickness = st.sidebar.slider("Толщина линии", 1, 3, 1)

    # --- АЛГОРИТМ БЕЗ ДВОЙНЫХ ЛИНИЙ ---
    
    # 1. Мягкое сглаживание для удаления микро-контраста
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Поиск основных границ (Canny)
    edges_main = cv2.Canny(smooth, edge_sensitivity // 2, edge_sensitivity)
    
    # 3. Адаптивный слой для теней (с защитой от дублирования)
    if shadow_detail > 0:
        # Используем Mean вместо Gaussian для более четких одиночных линий
        block_size = 3 + (shadow_detail * 2)
        soft_edges = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY, block_size, 5)
        soft_edges = cv2.bitwise_not(soft_edges)
        
        # Удаляем "шум" и тонкие двойные ореолы
        kernel_clean = np.ones((2,2), np.uint8)
        soft_edges = cv2.morphologyEx(soft_edges, cv2.MORPH_OPEN, kernel_clean)
        
        # Смешиваем, приоритет отдаем основным границам
        combined = cv2.bitwise_or(edges_main, soft_edges)
    else:
        combined = edges_main

    # 4. ФИНАЛЬНОЕ УТОНЧЕНИЕ (Убираем эффект дублирования)
    # Используем морфологический скелет для схлопывания близких линий
    if line_thickness == 1:
        # Тонкое сужение для удаления "двойного края"
        kernel_thin = np.ones((2,2), np.uint8)
        combined = cv2.erode(combined, kernel_thin, iterations=1)
        combined = cv2.Canny(combined, 50, 150) # Пересчитываем контур после сужения
    elif line_thickness > 1:
        kernel_thick = np.ones((line_thickness, line_thickness), np.uint8)
        combined = cv2.dilate(combined, kernel_thick, iterations=1)

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[combined > 0] = selected_color

    st.image(preview_img, caption="Чистый стенсил без дублирования", use_column_width=True)

    # PDF Функция
    def create_pdf(edge_data, width_cm, color_rgb):
        h, w = edge_data.shape
        aspect = h / w
        height_cm = width_cm * aspect
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        pdf_img_np = np.ones((h, w, 3), dtype=np.uint8) * 255
        pdf_img_np[edge_data > 0] = color_rgb
        temp_img = Image.fromarray(pdf_img_np)
        img_byte_arr = io.BytesIO()
        temp_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        from reportlab.lib.utils import ImageReader
        p.drawImage(ImageReader(img_byte_arr), 1*cm, (29.7 - height_cm - 1)*cm, width=width_cm*cm, height=height_cm*cm)
        p.showPage()
        p.save()
        return buffer.getvalue()

    pdf_data = create_pdf(combined, target_width_cm, selected_color)
    
    st.sidebar.markdown("---")
    st.sidebar.download_button(label="📥 Скачать PDF", data=pdf_data, file_name="angar_stencil.pdf", mime="application/pdf")
    
