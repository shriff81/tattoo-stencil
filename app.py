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
    
    # --- НАСТРОЙКИ В SIDEBAR ---
    st.sidebar.header("Параметры стенсила")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    
    # Выбор цвета
    color_name = st.sidebar.selectbox("Цвет линий:", ["Черный", "Ярко-красный", "Ярко-синий", "Ярко-зеленый"])
    colors_dict = {
        "Черный": [0, 0, 0],
        "Ярко-красный": [255, 0, 0],
        "Ярко-синий": [0, 0, 255],
        "Ярко-зеленый": [0, 255, 0]
    }
    sel_color = colors_dict[color_name]

    st.sidebar.subheader("Детализация и Очистка")
    detail_level = st.sidebar.slider("Детализация теней", 1, 15, 6)
    clean_level = st.sidebar.slider("Чистота (удаление точек)", 0, 50, 10)
    edge_thickness = st.sidebar.slider("Жирность контура", 1, 5, 1)
    
    st.sidebar.subheader("Просмотр")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)

    # --- АЛГОРИТМ ОБРАБОТКИ ---
    
    # 1. Основной контур (Canny)
    main_edges = cv2.Canny(gray, 100, 200)
    
    # 2. Линии переходов теней (Твоя логика уровней)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # Квантование яркости для поиска границ теней
    levels = np.floor(blur / (255 / detail_level)) * (255 / detail_level)
    shadow_edges = cv2.Canny(levels.astype(np.uint8), 10, 50)
    
    # Объединяем контуры
    combined_mask = cv2.bitwise_or(main_edges, shadow_edges)
    
    # 3. Удаление мелкого мусора (точек)
    if clean_level > 0:
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        refined_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
        for i in range(1, nb_components):
            if stats[i, cv2.CC_STAT_AREA] >= clean_level:
                refined_mask[output == i] = 255
        combined_mask = refined_mask

    # 4. Утолщение если нужно
    if edge_thickness > 1:
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel)

    # --- ВИЗУАЛИЗАЦИЯ (Preview) ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    
    # Создаем белый фон и накладываем оригинал
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_preview = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    # Рисуем линии выбранного цвета
    blended_preview[combined_mask > 0] = sel_color

    st.image(blended_preview, caption="Предпросмотр стенсила", use_column_width=True)

    # --- ГЕНЕРАЦИЯ PDF ---
    def create_pdf(mask_data, width_cm, color_rgb):
        h, w = mask_data.shape
        aspect = h / w
        height_cm = width_cm * aspect
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        
        # Для PDF создаем чистый слой (цветные линии на белом)
        pdf_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        pdf_img[mask_data > 0] = color_rgb
        
        temp_img = Image.fromarray(pdf_img)
        img_byte_arr = io.BytesIO()
        temp_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        from reportlab.lib.utils import ImageReader
        p.drawImage(ImageReader(img_byte_arr), 1*cm, (29.7 - height_cm - 1)*cm, width=width_cm*cm, height=height_cm*cm)
        p.showPage()
        p.save()
        return buffer.getvalue()

    pdf_data = create_pdf(combined_mask, target_width_cm, sel_color)
    
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label=f"📥 Скачать PDF ({color_name})",
        data=pdf_data,
        file_name="angar_stencil.pdf",
        mime="application/pdf"
    )
