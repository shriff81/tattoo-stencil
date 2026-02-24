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
    
    st.sidebar.header("Параметры стенсила")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    color_name = st.sidebar.selectbox("Цвет линий:", ["Черный", "Ярко-красный", "Ярко-синий", "Ярко-зеленый"])
    colors_dict = {"Черный": [0,0,0], "Ярко-красный": [255,0,0], "Ярко-синий": [0,0,255], "Ярко-зеленый": [0,255,0]}
    sel_color = colors_dict[color_name]

    st.sidebar.subheader("Структура и Детали")
    # Ваши базовые настройки
    detail_level = st.sidebar.slider("Основные уровни теней", 1, 15, 6)
    # НОВЫЙ ПОЛЗУНОК для "зеленых зон"
    soft_shadows = st.sidebar.slider("Добавить мягкие переходы (мышцы/крылья)", 0, 50, 0)
    
    st.sidebar.subheader("Очистка и Толщина")
    clean_level = st.sidebar.slider("Чистота (удаление точек)", 0, 100, 15)
    edge_thickness = st.sidebar.slider("Жирность линий", 1, 5, 1)
    
    st.sidebar.subheader("Просмотр")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)

    # --- АЛГОРИТМ ОБРАБОТКИ ---
    
    # 1. Основной контур (Резкие края)
    main_edges = cv2.Canny(gray, 100, 200)
    
    # 2. Изолинии уровней (Ваша базовая логика)
    # Используем сильное сглаживание, чтобы уровни были чистыми
    blur_base = cv2.bilateralFilter(gray, 9, 75, 75)
    levels = np.floor(blur_base / (255 / detail_level)) * (255 / detail_level)
    level_edges = cv2.Canny(levels.astype(np.uint8), 10, 50)
    
    # 3. НОВЫЙ СЛОЙ: Мягкие переходы (DoG)
    # Этот слой ищет то, что пропустили уровни
    if soft_shadows > 0:
        # Разница между сильным и слабым размытием выделяет средние детали
        g1 = cv2.GaussianBlur(gray, (3, 3), 0)
        g2 = cv2.GaussianBlur(gray, (21, 21), 0)
        dog = cv2.subtract(g2, g1)
        # Чем выше soft_shadows, тем больше линий проявится
        threshold_val = 55 - soft_shadows
        _, soft_edges = cv2.threshold(dog, threshold_val, 255, cv2.THRESH_BINARY)
    else:
        soft_edges = np.zeros_like(gray)

    # 4. Объединение всех слоев
    combined = cv2.bitwise_or(main_edges, level_edges)
    combined = cv2.bitwise_or(combined, soft_edges)
    
    # 5. Очистка от мусора (точек)
    if clean_level > 0:
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
        combined = np.zeros(combined.shape, dtype=np.uint8)
        for i in range(1, nb_components):
            if stats[i, cv2.CC_STAT_AREA] >= clean_level:
                combined[output == i] = 255

    # 6. Толщина
    if edge_thickness > 1:
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        combined = cv2.dilate(combined, kernel)

    # --- ПРЕДПРОСМОТР ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    blended[combined > 0] = sel_color

    st.image(blended, caption="Базовый алгоритм + Мягкие переходы", use_column_width=True)

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

    pdf = create_pdf(combined, target_width_cm, sel_color)
    st.sidebar.download_button("📥 Скачать PDF", data=pdf, file_name="angar_stencil.pdf")
