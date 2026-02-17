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
    
    st.sidebar.subheader("Логика отрисовки")
    # shadow_boost - отвечает за те самые "зеленые зоны"
    shadow_boost = st.sidebar.slider("Проработка анатомии (Shadows)", 1, 100, 50)
    # connectivity - склеивает точки в линии
    connectivity = st.sidebar.slider("Связность линий (убирает точки)", 1, 5, 2)
    # clean_level - удаляет мелкий мусор
    clean_level = st.sidebar.slider("Чистота (удаление мусора)", 1, 50, 15)

    # --- АЛГОРИТМ "MEDIAL AXIS" (ЦЕНТРАЛЬНАЯ ЛИНИЯ) ---

    # 1. Подготовка и усиление деталей
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    smooth = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # 2. Выделение зон теней (анатомии)
    block_size = 15
    # Константа регулирует "впитывание" теней
    const = (100 - shadow_boost) / 5
    binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, block_size, const)
    binary = cv2.bitwise_not(binary)

    # 3. СКЛЕИВАНИЕ ПУНКТИРА (Превращаем точки в линии)
    # Сначала расширяем, чтобы точки соприкоснулись
    kernel_conn = np.ones((connectivity, connectivity), np.uint8)
    dilated = cv2.dilate(binary, kernel_conn, iterations=1)
    
    # 4. УДАЛЕНИЕ МУСОРА
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    solid_lines = np.zeros(dilated.shape, dtype=np.uint8)
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= clean_level:
            solid_lines[output == i] = 255

    # 5. ИСТОНЧЕНИЕ (Skeletonization через Distance Transform)
    # Находим центры получившихся форм
    dist = cv2.distanceTransform(solid_lines, cv2.DIST_L2, 3)
    # Оставляем только "гребень" дистанции (самую середину линии)
    _, skeleton = cv2.threshold(dist, 0.4 * dist.max() if dist.max() > 0 else 0, 255, cv2.THRESH_BINARY)
    final_stencil = skeleton.astype(np.uint8)
    
    # Делаем финальный контур острым через Canny
    final_stencil = cv2.Canny(final_stencil, 50, 150)

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[final_stencil > 0] = selected_color

    st.image(preview_img, caption="Чистые анатомические линии (без точек и дублей)", use_column_width=True)

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

    pdf_data = create_pdf(final_stencil, target_width_cm, selected_color)
    st.sidebar.markdown("---")
    st.sidebar.download_button(label="📥 Скачать PDF", data=pdf_data, file_name="angar_stencil.pdf", mime="application/pdf")
    
