import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

st.set_page_config(page_title="AnGar Stencil Pro", layout="wide")
st.title("AnGar Stencil Pro 🎨")

uploaded_file = st.file_uploader("Загрузите референс", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    st.sidebar.header("Настройки Мастера")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    
    color_name = st.sidebar.selectbox("Цвет линий:", ["Черный", "Ярко-красный", "Ярко-синий", "Ярко-зеленый"])
    colors_dict = {"Черный": [0,0,0], "Ярко-красный": [255,0,0], "Ярко-синий": [0,0,255], "Ярко-зеленый": [0,255,0]}
    sel_color = colors_dict[color_name]

    st.sidebar.subheader("1. Топография (Основа)")
    num_levels = st.sidebar.slider("Количество уровней градации", 2, 15, 8)
    
    st.sidebar.subheader("2. Детализация (Тени/Лицо)")
    # Этот ползунок проявляет детали в глубоких тенях (глаза, губы, мышцы)
    shadow_boost = st.sidebar.slider("Проявление глубоких теней", 0, 255, 50)
    
    st.sidebar.subheader("3. Фильтр и Толщина")
    min_size_mm = st.sidebar.slider("Игнорировать детали меньше (мм)", 0.1, 3.0, 1.0, step=0.1)
    line_thickness = st.sidebar.slider("Жирность линии", 1, 4, 1)
    
    st.sidebar.subheader("Просмотр")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)

    # --- АЛГОРИТМ ОБРАБОТКИ ---
    
    # Подготовка: качественное сглаживание перед делением на зоны
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    # === СЛОЙ 1: ТОПОГРАФИЯ (Тонкие границы зон) ===
    # Делим изображение на четкие зоны яркости
    factor = 255 // (num_levels - 1)
    quantized = (smooth // factor) * factor
    # Применяем Canny к постеризованному изображению - это дает ТОНКИЕ границы
    topo_edges = cv2.Canny(quantized, 50, 150)

    # === СЛОЙ 2: ГЛУБОКИЕ ТЕНИ (Для глаз, губ и пропусков) ===
    # Выделяем самые темные участки
    _, dark_zones = cv2.threshold(smooth, shadow_boost, 255, cv2.THRESH_BINARY_INV)
    # Находим их тонкие границы
    shadow_edges = cv2.Canny(dark_zones, 50, 150)

    # === ОБЪЕДИНЕНИЕ ===
    combined = cv2.bitwise_or(topo_edges, shadow_edges)

    # === ОЧИСТКА МУСОРА ПО РАЗМЕРУ ===
    h_px, w_px = gray.shape
    px_per_mm = w_px / (target_width_cm * 10)
    min_area_px = (min_size_mm * px_per_mm) ** 2

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    cleaned_mask = np.zeros(combined.shape, dtype=np.uint8)
    
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned_mask[output == i] = 255

    # === ФИНАЛЬНАЯ ТОЛЩИНА ===
    if line_thickness > 1:
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        final_stencil = cv2.dilate(cleaned_mask, kernel)
    else:
        final_stencil = cleaned_mask

    # --- ПРЕДПРОСМОТР ---
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h_px, w_px, 3), dtype=np.uint8) * 255
    blended = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    blended[final_stencil > 0] = sel_color

    st.image(blended, caption=f"Стенсил: Топография + Тени", use_column_width=True)

    # --- PDF ---
    def create_pdf(mask, width, color_rgb):
        h, w = mask.shape
        aspect = h / w
        height = width * aspect
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        pdf_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        pdf_img[mask > 0] = color_rgb
        temp_img = Image.fromarray(pdf_img)
        img_io = io.BytesIO()
        temp_img.save(img_io, format='PNG')
        img_io.seek(0)
        from reportlab.lib.utils import ImageReader
        p.drawImage(ImageReader(img_io), 1*cm, (29.7 - height - 1)*cm, width=width*cm, height=height*cm)
        p.showPage()
        p.save()
        return buffer.getvalue()

    pdf = create_pdf(final_stencil, target_width_cm, sel_color)
    st.sidebar.markdown("---")
    st.sidebar.download_button("📥 Скачать PDF", data=pdf, file_name="angar_stencil.pdf")
