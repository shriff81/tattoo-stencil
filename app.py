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
    
    st.sidebar.subheader("Визуальный контроль")
    bg_opacity = st.sidebar.slider("Прозрачность оригинала (%)", 0, 100, 30)
    
    st.sidebar.subheader("Тонкая настройка линий")
    edge_sensitivity = st.sidebar.slider("Чувствительность контуров", 10, 250, 120)

    # 1. Генерация стенсила
    smooth = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(smooth, edge_sensitivity // 2, edge_sensitivity)

    # 2. Создание превью на БЕЛОМ фоне
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    
    # Создаем чисто белый холст
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Смешиваем оригинал с белым фоном (эффект высветления)
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    # Накладываем красные линии поверх высветленного оригинала
    preview_img = blended_bg.copy()
    preview_img[edges > 0] = [255, 0, 0]

    st.image(preview_img, caption="Контроль наложения (на белом фоне)", use_column_width=True)

    # Функция PDF (чистый красный на белом)
    def create_pdf(edge_data, width_cm):
        h, w = edge_data.shape
        aspect = h / w
        height_cm = width_cm * aspect
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        
        pdf_img_np = np.ones((h, w, 3), dtype=np.uint8) * 255
        pdf_img_np[edge_data > 0] = [255, 0, 0]
        
        temp_img = Image.fromarray(pdf_img_np)
        img_byte_arr = io.BytesIO()
        temp_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        from reportlab.lib.utils import ImageReader
        p.drawImage(ImageReader(img_byte_arr), 1*cm, (29.7 - height_cm - 1)*cm, width=width_cm*cm, height=height_cm*cm)
        p.showPage()
        p.save()
        return buffer.getvalue()

    pdf_data = create_pdf(edges, target_width_cm)
    
    st.download_button(
        label="📥 Скачать ЧИСТЫЙ КРАСНЫЙ PDF (1:1)",
        data=pdf_data,
        file_name="angar_red_stencil.pdf",
        mime="application/pdf"
    )
