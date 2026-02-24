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

# --- ФУНКЦИЯ СКЕЛЕТИЗАЦИИ (ДЛЯ ТОНКИХ ЛИНИЙ) ---
def get_thin_stencil(binary_img):
    # Убираем шум перед истончением
    kernel = np.ones((3,3), np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    
    # Алгоритм скелетизации (Zhang-Suen упрощенно)
    skel = np.zeros(binary_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    temp = binary_img.copy()
    
    while True:
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, dilated)
        skel = cv2.bitwise_or(skel, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break
    return skel

uploaded_file = st.file_uploader("Загрузите референс", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    st.sidebar.header("Настройки Tattoo Stencil Pro")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    
    color_name = st.sidebar.selectbox("Цвет линий:", ["Черный", "Ярко-красный", "Ярко-синий", "Ярко-зеленый"])
    colors_dict = {"Черный": [0,0,0], "Ярко-красный": [255,0,0], "Ярко-синий": [0,0,255], "Ярко-зеленый": [0,255,0]}
    sel_color = colors_dict[color_name]

    st.sidebar.subheader("1. Градация (Topography)")
    num_levels = st.sidebar.slider("Количество оттенков (градация)", 2, 12, 6)
    
    st.sidebar.subheader("2. Детализация")
    # shadow_boost вытягивает тени в пропущенных зонах (глаза, губы)
    shadow_boost = st.sidebar.slider("Глубина теней (Shadow Boost)", 0, 100, 40)
    
    st.sidebar.subheader("3. Фильтр мусора")
    min_size_mm = st.sidebar.slider("Игнорировать детали меньше (мм)", 0.1, 5.0, 1.2, step=0.1)
    
    st.sidebar.subheader("Просмотр")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)

    # --- АЛГОРИТМ "PRO STENCIL" ---
    
    # А. Подготовка: Усиливаем детали в тенях (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    smooth = cv2.bilateralFilter(cl, 9, 75, 75)

    # Б. Постеризация (Твоя идея: Деление на N оттенков)
    factor = 255 // (num_levels - 1)
    quantized = (smooth // factor) * factor
    
    # В. Поиск границ зон + Глубокие тени (Shadow Boost)
    # Находим границы между цветовыми зонами
    topo_edges = cv2.morphologyEx(quantized, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    
    # Дополнительно обводим очень темные участки (глаза, губы)
    _, black_mask = cv2.threshold(cl, 255 - shadow_boost, 255, cv2.THRESH_BINARY_INV)
    black_edges = cv2.morphologyEx(black_mask, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    
    combined = cv2.bitwise_or(topo_edges, black_edges)
    _, binary_stencil = cv2.threshold(combined, 1, 255, cv2.THRESH_BINARY)

    # Г. Очистка мусора по физическому размеру (мм)
    px_per_mm = gray.shape[1] / (target_width_cm * 10)
    min_area_px = (min_size_mm * px_per_mm) ** 2

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(binary_stencil, connectivity=8)
    cleaned = np.zeros(binary_stencil.shape, dtype=np.uint8)
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned[output == i] = 255

    # Д. ФИНАЛЬНОЕ УТОНЧЕНИЕ (Skeletonization)
    # Это превращает любую границу в идеально тонкую линию (как на сайте)
    final_stencil = get_thin_stencil(cleaned)

    # --- ПРЕДПРОСМОТР ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    blended[final_stencil > 0] = sel_color

    st.image(blended, caption="Результат: Чистые анатомические линии", use_column_width=True)

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
    st.sidebar.download_button("📥 Скачать PDF для печати", data=pdf, file_name="angar_stencil.pdf")
