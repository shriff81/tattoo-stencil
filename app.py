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
    
    st.sidebar.header("Настройки Мастера")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    
    stencil_color_name = st.sidebar.selectbox(
        "Цвет стенсила:", ["Ярко-красный", "Ярко-синий", "Ярко-зеленый", "Черный"]
    )
    colors_dict = {"Ярко-красный": [255, 0, 0], "Ярко-синий": [0, 0, 255], "Ярко-зеленый": [0, 255, 0], "Черный": [0, 0, 0]}
    selected_color = colors_dict[stencil_color_name]

    st.sidebar.subheader("Визуальный контроль")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 40)
    
    st.sidebar.subheader("Логика ручной отрисовки")
    # shadow_sense - вытягивает те самые зеленые зоны анатомии
    shadow_sense = st.sidebar.slider("Детализация анатомии", 1, 100, 45)
    # clean_level - удаляет те самые "точки"
    clean_level = st.sidebar.slider("Чистка от мусора (точек)", 1, 50, 15)
    # line_weight - делает линию увереннее
    line_weight = st.sidebar.slider("Толщина линии", 1, 3, 1)

    # --- АЛГОРИТМ "СВЯЗНЫЙ КОНТУР" ---

    # А. Подготовка и усиление теней
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    smooth = cv2.bilateralFilter(cl, 9, 75, 75)

    # Б. Выделение "хребтов" теней
    block_size = 13
    constant = (100 - shadow_sense) / 5
    binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, block_size, constant)
    lines = cv2.bitwise_not(binary)

    # В. СКЛЕИВАНИЕ ТОЧЕК В ЛИНИИ
    # Используем морфологию для соединения разрывов
    kernel_connect = np.ones((2,2), np.uint8)
    connected = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel_connect)

    # Г. УДАЛЕНИЕ МЕЛКИХ ОБЪЕКТОВ (ТОЧЕК)
    # Находим все отдельные элементы
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)
    # Создаем пустой холст
    cleaned_lines = np.zeros(connected.shape, dtype=np.uint8)
    
    # Оставляем только те элементы, которые достаточно длинные/большие
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= clean_level:
            cleaned_lines[output == i] = 255

    # Д. ФИНАЛЬНОЕ УТОНЧЕНИЕ
    # Чтобы линии не были жирными, но и не превращались в точки
    final_stencil = cv2.erode(cleaned_lines, np.ones((2,2), np.uint8), iterations=1)
    final_stencil = cv2.Canny(final_stencil, 50, 150)

    if line_weight > 1:
        final_stencil = cv2.dilate(final_stencil, np.ones((line_weight, line_weight), np.uint8))

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[final_stencil > 0] = selected_color

    st.image(preview_img, caption="Стенсил без мусора и точек", use_column_width=True)

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
    
