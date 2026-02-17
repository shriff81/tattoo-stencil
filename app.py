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
    # Увеличиваем этот параметр, чтобы найти потерянные тени
    shadow_depth = st.sidebar.slider("Глубина поиска теней", 1, 50, 25)
    edge_force = st.sidebar.slider("Четкость контура", 10, 250, 150)
    line_thickness = st.sidebar.slider("Толщина линии", 1, 3, 1)

    # --- НОВЫЙ АЛГОРИТМ: СХЛОПЫВАНИЕ ДВОЙНЫХ ЛИНИЙ + ТЕНИ ---
    
    # 1. Выравнивание освещения для проявления деталей
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    
    # 2. Поиск границ с защитой от разрывов
    blur = cv2.GaussianBlur(cl, (5, 5), 0)
    edges = cv2.Canny(blur, edge_force // 2, edge_force)
    
    # 3. Выделение зон теней (обводка элементов)
    shadow_mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 21, shadow_depth // 5)
    shadow_lines = cv2.Canny(shadow_mask, 10, 50)
    
    # Объединяем слои
    combined = cv2.bitwise_or(edges, shadow_lines)
    
    # 4. ФИНАЛЬНОЕ РЕШЕНИЕ ПРОБЛЕМЫ ДУБЛИРОВАНИЯ (Скелетизация)
    # Мы превращаем любую область в линию толщиной 1 пиксель
    kernel = np.ones((3,3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel) # Соединяем разрывы
    
    # Используем Zhang-Suen алгоритм для схлопывания двойных линий в одну
    skeleton = cv2.ximgproc.thinning(combined)

    if line_thickness > 1:
        kernel_thick = np.ones((line_thickness, line_thickness), np.uint8)
        final_edges = cv2.dilate(skeleton, kernel_thick, iterations=1)
    else:
        final_edges = skeleton

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[final_edges > 0] = selected_color

    st.image(preview_img, caption="Исправленный стенсил: одна линия и глубокие тени", use_column_width=True)

    # Функция PDF
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

    pdf_data = create_pdf(final_edges, target_width_cm, selected_color)
    st.sidebar.download_button(label="📥 Скачать PDF", data=pdf_data, file_name="angar_stencil.pdf", mime="application/pdf")
    
