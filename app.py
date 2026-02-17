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
    # Загрузка
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    st.sidebar.header("Настройки Мастера")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    
    # НОВЫЙ ВЫБОР ЦВЕТА
    stencil_color_name = st.sidebar.selectbox(
        "Цвет стенсила:",
        ["Ярко-красный", "Ярко-синий", "Ярко-зеленый", "Черный"]
    )
    
    # Словарь цветов (RGB)
    colors_dict = {
        "Ярко-красный": [255, 0, 0],
        "Ярко-синий": [0, 0, 255],
        "Ярко-зеленый": [0, 255, 0],
        "Черный": [0, 0, 0]
    }
    selected_color = colors_dict[stencil_color_name]

    st.sidebar.subheader("Визуальный контроль")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)
    
    st.sidebar.subheader("Тонкая настройка линий")
    edge_sensitivity = st.sidebar.slider("Чувствительность контуров", 10, 250, 120)
    line_thickness = st.sidebar.slider("Толщина линии", 1, 5, 1)

    # 1. Генерация базового контура
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(smooth, edge_sensitivity // 2, edge_sensitivity)

    # 2. Плавное утолщение линий
    if line_thickness > 1:
        kernel = np.ones((line_thickness, line_thickness), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    # 3. Создание превью на БЕЛОМ фоне
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[edges > 0] = selected_color # Накладываем выбранный цвет

    st.image(preview_img, caption=f"Стенсил: {stencil_color_name}", use_column_width=True)

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

    pdf_data = create_pdf(edges, target_width_cm, selected_color)
    
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label=f"📥 Скачать PDF ({stencil_color_name})",
        data=pdf_data,
        file_name=f"angar_stencil_{stencil_color_name}.pdf",
        mime="application/pdf"
    )
    
