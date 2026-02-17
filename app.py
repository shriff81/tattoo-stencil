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
    # 1. Загрузка и подготовка
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    st.sidebar.header("Настройки Мастера")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    
    stencil_color_name = st.sidebar.selectbox(
        "Цвет стенсила:", ["Ярко-красный", "Ярко-синий", "Ярко-зеленый", "Черный"]
    )
    colors_dict = {"Ярко-красный": [255, 0, 0], "Ярко-синий": [0, 0, 255], "Ярко-зеленый": [0, 255, 0], "Черный": [0, 0, 0]}
    selected_color = colors_dict[stencil_color_name]

    st.sidebar.subheader("Визуальный контроль")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 40)
    
    st.sidebar.subheader("Проработка анатомии")
    # shadow_boost - вытягивает те самые зеленые линии
    shadow_boost = st.sidebar.slider("Проявление теней", 1, 100, 50)
    noise_filter = st.sidebar.slider("Чистота (удаление пор)", 1, 15, 5)
    line_boldness = st.sidebar.slider("Жирность линий", 1, 3, 1)

    # --- АЛГОРИТМ "ЦЕНТРАЛЬНОЙ ЛИНИИ" ---

    # А. Усиление теней через CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)

    # Б. Фильтрация пор кожи
    smooth = cv2.bilateralFilter(cl, 9, noise_filter * 10, noise_filter * 10)

    # В. Поиск "хребтов" теней (Ridge Detection упрощенно)
    # Используем адаптивный порог с маленьким окном для поиска тонких линий
    block_size = 7
    c_val = 101 - shadow_boost
    ridges = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, block_size, c_val / 10)
    ridges = cv2.bitwise_not(ridges)

    # Г. Основной контур (Canny) для четких краев
    edges = cv2.Canny(smooth, 100, 200)

    # Д. Объединение и схлопывание двойных линий
    combined = cv2.bitwise_or(edges, ridges)
    
    # Финальная очистка и утончение
    kernel = np.ones((2,2), np.uint8)
    # Убираем одиночные пиксели (мусор)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Скелетизация (оставляем только центр линии)
    final_stencil = cv2.erode(cleaned, kernel, iterations=1)
    final_stencil = cv2.Canny(final_stencil, 50, 150)

    if line_boldness > 1:
        final_stencil = cv2.dilate(final_stencil, np.ones((line_boldness, line_boldness), np.uint8))

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[final_stencil > 0] = selected_color

    st.image(preview_img, caption="Стенсил готов (Центральные линии)", use_column_width=True)

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

    pdf_data = create_pdf(final_stencil, target_width_cm, selected_color)
    st.sidebar.download_button(label="📥 Скачать PDF", data=pdf_data, file_name="angar_stencil.pdf", mime="application/pdf")
    
