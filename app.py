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
    # 1. Загрузка и подготовка
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    st.sidebar.header("Параметры стенсила")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    
    # Цвет стенсила
    color_name = st.sidebar.selectbox("Цвет линий:", ["Черный", "Ярко-красный", "Ярко-синий", "Ярко-зеленый"])
    colors_dict = {"Черный": [0,0,0], "Ярко-красный": [255,0,0], "Ярко-синий": [0,0,255], "Ярко-зеленый": [0,255,0]}
    sel_color = colors_dict[color_name]

    st.sidebar.subheader("Детализация")
    # CLAHE вытянет тени в зеленых зонах
    contrast_boost = st.sidebar.slider("Усиление деталей (Contrast)", 1.0, 10.0, 3.0)
    detail_level = st.sidebar.slider("Слои теней", 1, 15, 6)
    
    st.sidebar.subheader("Очистка и Толщина")
    clean_level = st.sidebar.slider("Чистота (удаление точек)", 0, 100, 15)
    edge_thickness = st.sidebar.slider("Жирность линий", 1, 5, 1)
    
    st.sidebar.subheader("Просмотр")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)

    # --- АЛГОРИТМ ОБРАБОТКИ ---
    
    # А. Усиление локального контраста (CLAHE) - решение для зеленых зон
    clahe = cv2.createCLAHE(clipLimit=contrast_boost, tileGridSize=(8,8))
    cl = clahe.apply(gray)

    # Б. Твоя логика изолиний
    blur = cv2.bilateralFilter(cl, 9, 75, 75)
    levels = np.floor(blur / (255 / detail_level)) * (255 / detail_level)
    shadow_edges = cv2.Canny(levels.astype(np.uint8), 10, 50)
    
    # В. Основной контур
    main_edges = cv2.Canny(cl, 100, 200)
    
    # Г. Объединение
    combined = cv2.bitwise_or(main_edges, shadow_edges)
    
    # Д. Очистка от мусора (точек)
    if clean_level > 0:
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
        combined = np.zeros(combined.shape, dtype=np.uint8)
        for i in range(1, nb_components):
            if stats[i, cv2.CC_STAT_AREA] >= clean_level:
                combined[output == i] = 255

    # Е. Толщина
    if edge_thickness > 1:
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        combined = cv2.dilate(combined, kernel)

    # --- ПРЕДПРОСМОТР ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    blended[combined > 0] = sel_color

    st.image(blended, caption="Обновленный стенсил", use_column_width=True)

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
