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
    # 1. Загрузка
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    st.sidebar.header("Параметры Мастера")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    
    color_name = st.sidebar.selectbox("Цвет линий:", ["Черный", "Ярко-красный", "Ярко-синий", "Ярко-зеленый"])
    colors_dict = {"Черный": [0,0,0], "Ярко-красный": [255,0,0], "Ярко-синий": [0,0,255], "Ярко-зеленый": [0,255,0]}
    sel_color = colors_dict[color_name]

    st.sidebar.subheader("1. Базовая структура (Топография)")
    num_levels = st.sidebar.slider("Количество уровней градации", 2, 12, 6)
    
    st.sidebar.subheader("2. Дополнительные детали")
    # Этот ползунок вытаскивает линии в "зеленых зонах"
    anatomy_boost = st.sidebar.slider("Усиление анатомии (мышцы/перья)", 0, 100, 30)
    
    st.sidebar.subheader("3. Финальная очистка")
    min_size_mm = st.sidebar.slider("Игнорировать мусор меньше (мм)", 0.1, 3.0, 1.0, step=0.1)
    
    st.sidebar.subheader("Просмотр")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)

    # --- АЛГОРИТМ ОБРАБОТКИ ---
    
    # Подготовка: качественное сглаживание
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    # === СЛОЙ 1: ТОПОГРАФИЯ (Градация) ===
    factor = 255 // (num_levels - 1)
    quantized = (smooth // factor) * factor
    kernel_grad = np.ones((3,3), np.uint8)
    topo_edges = cv2.morphologyEx(quantized, cv2.MORPH_GRADIENT, kernel_grad)
    _, topo_edges = cv2.threshold(topo_edges, 1, 255, cv2.THRESH_BINARY)

    # === СЛОЙ 2: АНАТОМИЯ (DoG для пропущенных деталей) ===
    if anatomy_boost > 0:
        # Ищем детали среднего размера (мышцы, перья)
        g1 = cv2.GaussianBlur(gray, (3, 3), 0)
        g2 = cv2.GaussianBlur(gray, (15, 15), 0)
        dog = cv2.subtract(g2, g1)
        # Чем выше anatomy_boost, тем больше деталей проявляется
        threshold_val = 50 - (anatomy_boost // 2)
        threshold_val = max(5, threshold_val) # Защита от слишком низкого порога
        _, anatomy_edges = cv2.threshold(dog, threshold_val, 255, cv2.THRESH_BINARY)
    else:
        anatomy_edges = np.zeros_like(gray)

    # === ОБЪЕДИНЕНИЕ СЛОЕВ ===
    combined = cv2.bitwise_or(topo_edges, anatomy_edges)

    # === ФИНАЛЬНЫЙ ФИЛЬТР МУСОРА ПО РАЗМЕРУ (в мм) ===
    h_px, w_px = gray.shape
    px_per_mm = w_px / (target_width_cm * 10)
    min_area_px = (min_size_mm * px_per_mm) ** 2

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    final_stencil = np.zeros(combined.shape, dtype=np.uint8)
    
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            final_stencil[output == i] = 255

    # --- ПРЕДПРОСМОТР ---
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h_px, w_px, 3), dtype=np.uint8) * 255
    blended = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    blended[final_stencil > 0] = sel_color

    st.image(blended, caption=f"Гибридный стенсил: Градация + Анатомия", use_column_width=True)

    # --- ГЕНЕРАЦИЯ PDF ---
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
