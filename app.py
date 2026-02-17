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
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)
    
    st.sidebar.subheader("Сила анатомических линий")
    # shadow_power - теперь это "магнит" для тех самых зеленых линий
    shadow_power = st.sidebar.slider("Проработка мягких переходов", 1, 100, 45)
    noise_filter = st.sidebar.slider("Чистота (удаление пор)", 1, 15, 5)
    line_thickness = st.sidebar.slider("Толщина линии", 1, 3, 1)

    # --- АЛГОРИТМ RIDGE DETECTION ---

    # 1. Усиление локальных деталей
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)

    # 2. Удаление пор (Bilateral filter)
    smooth = cv2.bilateralFilter(cl, 9, noise_filter * 10, noise_filter * 10)

    # 3. Метод "Разности Гауссиан" (DoG) для поиска скрытых линий
    # Это вытаскивает тени, которые вы пометили зеленым
    g1 = cv2.GaussianBlur(smooth, (3, 3), 0)
    g2 = cv2.GaussianBlur(smooth, (9, 9), 0)
    dog = cv2.subtract(g1, g2)
    
    # Порог для проявления линий
    _, anatomical_lines = cv2.threshold(dog, 255 - shadow_power * 2.5, 255, cv2.THRESH_BINARY)

    # 4. Схлопывание и утончение
    kernel = np.ones((2,2), np.uint8)
    # Соединяем микро-разрывы
    connected = cv2.morphologyEx(anatomical_lines, cv2.MORPH_CLOSE, kernel)
    
    # Принудительно оставляем только тонкий центр (Skeletonization)
    skeleton = np.zeros(connected.shape, np.uint8)
    temp = connected.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    while True:
        eroded = cv2.erode(temp, element)
        dilated = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, dilated)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    # Финальная толщина
    if line_thickness > 1:
        final_edges = cv2.dilate(skeleton, np.ones((line_thickness, line_thickness), np.uint8))
    else:
        final_edges = skeleton

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[final_edges > 0] = selected_color

    st.image(preview_img, caption="Обновленный стенсил: Анатомические линии + Чистота", use_column_width=True)

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

    pdf_data = create_pdf(final_edges, target_width_cm, selected_color)
    st.sidebar.download_button(label="📥 Скачать PDF", data=pdf_data, file_name="angar_stencil.pdf", mime="application/pdf")
    
