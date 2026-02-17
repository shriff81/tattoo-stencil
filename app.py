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
    
    st.sidebar.subheader("Геометрия линий")
    # shadow_detail теперь работает точечно, не создавая жирности
    shadow_detail = st.sidebar.slider("Проработка теней", 0, 10, 2)
    noise_reduction = st.sidebar.slider("Чистота (удаление пор)", 1, 10, 3)
    edge_sensitivity = st.sidebar.slider("Чувствительность", 10, 250, 150)

    # --- АЛГОРИТМ ЧИСТОГО КОНТУРА ---
    
    # 1. Удаление пор и шума (Сильный Bilateral Filter)
    smooth = cv2.bilateralFilter(gray, 9, noise_reduction * 15, noise_reduction * 15)
    
    # 2. Поиск границ (Canny)
    edges = cv2.Canny(smooth, edge_sensitivity // 2, edge_sensitivity)
    
    # 3. Добавление деталей теней без дублирования
    if shadow_detail > 0:
        # Используем Laplacian для поиска центра теней
        laplacian = cv2.Laplacian(smooth, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, shadow_edges = cv2.threshold(laplacian, 255 - (shadow_detail * 20), 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(edges, shadow_edges)
    else:
        combined = edges

    # 4. ФИНАЛЬНОЕ УТОНЧЕНИЕ (Skeletonization)
    # Схлопываем любые намеки на двойные линии в одну центральную трассу
    kernel = np.ones((2,2), np.uint8)
    final_edges = cv2.morphologyEx(combined, cv2.MORPH_ERODE, kernel)
    final_edges = cv2.Canny(final_edges, 100, 200) # Оставляем только финальный резкий пиксель

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[final_edges > 0] = selected_color

    st.image(preview_img, caption="Результат: Чистый стенсил", use_column_width=True)

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
    
    st.sidebar.markdown("---")
    st.sidebar.download_button(label="📥 Скачать PDF", data=pdf_data, file_name="angar_stencil.pdf", mime="application/pdf")
    
