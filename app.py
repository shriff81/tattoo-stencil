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
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Настройки
    st.sidebar.header("Параметры стенсила")
    target_width_cm = st.sidebar.number_input("Ширина печати (см)", 5.0, 30.0, 15.0)
    detail_level = st.sidebar.slider("Детализация теней (линии переходов)", 1, 10, 4)
    edge_thickness = st.sidebar.slider("Жирность контура", 1, 5, 1)

    # Алгоритм: Чистые линии без серого
    # 1. Основной контур
    main_edges = cv2.Canny(gray, 100, 200)
    
    # 2. Линии переходов теней (изолинии)
    # Используем аппроксимацию уровней для создания четких границ между тонами
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    levels = np.floor(blur / (255 / detail_level)) * (255 / detail_level)
    shadow_edges = cv2.Canny(levels.astype(np.uint8), 10, 50)
    
    # Объединяем всё в один чисто черный слой на белом фоне
    final_stencil = np.ones_like(gray) * 255
    final_stencil[shadow_edges > 0] = 0
    final_stencil[main_edges > 0] = 0
    
    # Утолщение линий если нужно
    if edge_thickness > 1:
        kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
        final_stencil = cv2.erode(final_stencil, kernel)

    st.image(final_stencil, caption="Результат: Только линии", use_column_width=True)

    # Генерация PDF с точным размером
    def create_pdf(img_data, width_cm):
        # Сохраняем пропорции
        h, w = img_data.shape
        aspect = h / w
        height_cm = width_cm * aspect
        
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        
        # Превращаем массив в картинку для PDF
        temp_img = Image.fromarray(img_data)
        img_byte_arr = io.BytesIO()
        temp_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        from reportlab.lib.utils import ImageReader
        reportlab_img = ImageReader(img_byte_arr)
        
        # Рисуем в центре листа А4
        p.drawImage(reportlab_img, 1*cm, (29.7 - height_cm - 1)*cm, width=width_cm*cm, height=height_cm*cm)
        p.showPage()
        p.save()
        return buffer.getvalue()

    pdf_data = create_pdf(final_stencil, target_width_cm)
    
    st.download_button(
        label="Скачать PDF для печати (Точный размер)",
        data=pdf_data,
        file_name="angar_stencil.pdf",
        mime="application/pdf"
    )
