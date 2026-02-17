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
    # 1. Загрузка и первичная обработка
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
    # Анатомия - вытягивает те самые зеленые зоны
    anatomy_boost = st.sidebar.slider("Проработка анатомии", 1, 100, 40)
    # Чистота - убирает поры кожи
    noise_clean = st.sidebar.slider("Чистота (удаление текстуры)", 1, 20, 8)
    # Острота - контролирует толщину
    line_sharpness = st.sidebar.slider("Острота линий", 1, 5, 1)

    # --- АЛГОРИТМ "АНАТОМИЧЕСКИЙ ГРАДИЕНТ" ---

    # А. Усиление деталей в тенях
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)

    # Б. Сглаживание пор и шума
    smooth = cv2.bilateralFilter(cl, 9, noise_clean * 10, noise_clean * 10)

    # В. Извлечение "хребтов" теней (Ridge Detection)
    # Используем разницу размытий (DoG) для поиска центральных осей теней
    g1 = cv2.GaussianBlur(smooth, (3, 3), 0)
    g2 = cv2.GaussianBlur(smooth, (15, 15), 0)
    dog = cv2.subtract(g1, g2)
    
    # Порог проявления линий
    _, ridges = cv2.threshold(dog, 255 - (anatomy_boost * 2), 255, cv2.THRESH_BINARY)

    # Г. Устранение двойных линий и соединение разрывов
    kernel = np.ones((2,2), np.uint8)
    # Соединяем разрывы (Closing)
    closed = cv2.morphologyEx(ridges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Безопасное утончение (не удаляет линии целиком)
    skeleton = cv2.erode(closed, kernel, iterations=1)
    # Финальный контур через Canny для идеальной остроты
    final_stencil = cv2.Canny(skeleton, 50, 150)

    # Д. Регулировка толщины
    if line_sharpness > 1:
        final_stencil = cv2.dilate(final_stencil, np.ones((line_sharpness, line_sharpness), np.uint8))

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[final_stencil > 0] = selected_color

    st.image(preview_img, caption="Стенсил по твоей ручной логике", use_column_width=True)

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
    
