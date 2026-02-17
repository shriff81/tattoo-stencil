import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="AnGar Stencil Pro", layout="wide")
st.title("AnGar Stencil Pro 🎨")

uploaded_file = st.file_uploader("Загрузите фото", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Настройки
    st.sidebar.header("Настройки мастера")
    target_width_cm = st.sidebar.number_input("Ширина тату (см)", 5.0, 50.0, 15.0)
    main_edge = st.sidebar.slider("Четкость основных линий", 50, 200, 120)
    shadow_detail = st.sidebar.slider("Детализация теней", 1, 5, 3)

    # 1. Сплошные основные контуры (Canny)
    edges = cv2.Canny(gray, main_edge//2, main_edge)
    
    # 2. Создаем карту градиентов (Пунктир/Тонкие линии)
    # Используем адаптивный порог для выделения зон теней
    shadows = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, shadow_detail * 2)
    
    # Инвертируем и чистим шум
    shadows_inv = cv2.bitwise_not(shadows)
    kernel = np.ones((2,2), np.uint8)
    shadows_inv = cv2.morphologyEx(shadows_inv, cv2.MORPH_OPEN, kernel)

    # Смешиваем результат
    canvas = np.ones_like(gray) * 255
    # Накладываем тени серым цветом (имитация пунктира/тонкой линии)
    canvas[shadows_inv > 0] = 180
    # Накладываем основные контуры черным
    canvas[edges > 0] = 0

    st.image(canvas, caption="Готовый трансфер", use_column_width=True)
    
    # Подготовка файла
    res_img = Image.fromarray(canvas)
    buf = io.BytesIO()
    res_img.save(buf, format="PNG")
    
    st.download_button("Скачать stencil для печати", buf.getvalue(), "stencil.png", "image/png")
