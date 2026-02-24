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
        "Цвет стенсила:", ["Ярко-красный", "Ярко-синий", "Ярко-зеленый", "Черный"]
    )
    colors_dict = {"Ярко-красный": [255, 0, 0], "Ярко-синий": [0, 0, 255], "Ярко-зеленый": [0, 255, 0], "Черный": [0, 0, 0]}
    selected_color = colors_dict[stencil_color_name]

    st.sidebar.subheader("Визуальный контроль")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 40)
    
    st.sidebar.subheader("Фильтрация (Назад к деталям)")
    # Позволяет вытащить те самые "зеленые зоны"
    detail_boost = st.sidebar.slider("Детализация теней", 1, 100, 50)
    # Помогает слить двойные линии в одну
    line_fusion = st.sidebar.slider("Слияние линий (убирает двойные)", 1, 5, 1)
    # Убирает мелкую пыль
    noise_clean = st.sidebar.slider("Чистота (удаление шума)", 1, 50, 10)

    # --- АЛГОРИТМ "ПЛОТНЫЙ КОНТУР" ---

    # 1. Усиление контраста
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    
    # 2. Мягкое сглаживание
    smooth = cv2.bilateralFilter(cl, 9, 75, 75)

    # 3. Адаптивный порог (Дает много стенсила)
    block_size = 11
    # Чем меньше shadow_boost, тем больше деталей
    constant = (105 - detail_boost) / 5
    binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, block_size, constant)
    lines = cv2.bitwise_not(binary)

    # 4. БЕЗОПАСНАЯ ОЧИСТКА
    # Сначала сливаем близкие линии (решает проблему двойного контура)
    if line_fusion > 1:
        kernel_fuse = np.ones((line_fusion, line_fusion), np.uint8)
        lines = cv2.dilate(lines, kernel_fuse, iterations=1)
        lines = cv2.erode(lines, kernel_fuse, iterations=1)

    # 5. УДАЛЕНИЕ МУСОРА (ТОЧЕК)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    final_stencil = np.zeros(lines.shape, dtype=np.uint8)
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= noise_clean:
            final_stencil[output == i] = 255

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[final_stencil > 0] = selected_color

    st.image(preview_img, caption="Стенсил: Возврат к деталям", use_column_width=True)

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
    st.sidebar.markdown("---")
    st.sidebar.download_button(label="📥 Скачать PDF", data=pdf_data, file_name="angar_stencil.pdf", mime="application/pdf")
    
