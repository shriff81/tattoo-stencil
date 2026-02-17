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
        "Цвет стенсила:",
        ["Ярко-красный", "Ярко-синий", "Ярко-зеленый", "Черный"]
    )
    
    colors_dict = {
        "Ярко-красный": [255, 0, 0], "Ярко-синий": [0, 0, 255],
        "Ярко-зеленый": [0, 255, 0], "Черный": [0, 0, 0]
    }
    selected_color = colors_dict[stencil_color_name]

    st.sidebar.subheader("Визуальный контроль")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)
    
    st.sidebar.subheader("Настройка линий")
    # shadow_detail теперь добавляет линии там, где их не было
    shadow_detail = st.sidebar.slider("Добавить детали теней", 0, 10, 3)
    # noise_reduction убирает поры
    noise_reduction = st.sidebar.slider("Чистота (удаление пор)", 1, 10, 3)
    edge_sensitivity = st.sidebar.slider("Чувствительность", 10, 250, 130)

    # --- АЛГОРИТМ ЧИСТЫХ ЛИНИЙ ---
    
    # 1. Сглаживание пор (очень важно для реализма)
    # Bilateral filter сохраняет границы, но размывает кожу
    smooth = cv2.bilateralFilter(gray, 9, noise_reduction * 15, noise_reduction * 15)
    
    # 2. Основной контур (Canny - он лучше всего держит одну линию)
    edges = cv2.Canny(smooth, edge_sensitivity // 2, edge_sensitivity)
    
    # 3. Достаем потерянные тени
    if shadow_detail > 0:
        # Ищем области с мягким перепадом яркости
        grad_x = cv2.Sobel(smooth, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(smooth, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # Обводим только значимые переходы теней
        _, shadow_lines = cv2.threshold(grad, 255 - (shadow_detail * 20), 255, cv2.THRESH_BINARY)
        final_stencil = cv2.bitwise_or(edges, shadow_lines)
    else:
        final_stencil = edges

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[final_stencil > 0] = selected_color

    st.image(preview_img, caption="Стенсил готов", use_column_width=True)

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
    
