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
    # 1. Подготовка изображения
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
    
    st.sidebar.subheader("Проработка деталей")
    # shadow_boost - вытягивает те самые зеленые линии из вашего скриншота
    shadow_boost = st.sidebar.slider("Усиление скрытых теней", 1, 50, 25)
    noise_clean = st.sidebar.slider("Чистота (удаление пор)", 1, 10, 3)
    edge_sens = st.sidebar.slider("Чувствительность контуров", 10, 250, 140)

    # --- ПРОФЕССИОНАЛЬНЫЙ АЛГОРИТМ КОНТУРОВ ---

    # А. Усиливаем локальный контраст (чтобы видеть детали в тенях)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)

    # Б. Удаляем шум, сохраняя резкость
    smooth = cv2.bilateralFilter(cl, 9, 75, 75)
    denoised = cv2.medianBlur(smooth, (noise_clean * 2 - 1) if noise_clean > 0 else 1)

    # В. Извлекаем "зеленые зоны" (тени и узкие детали) через Black Hat
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    _, shadow_details = cv2.threshold(blackhat, 51 - (shadow_boost), 255, cv2.THRESH_BINARY)

    # Г. Основной контур
    edges = cv2.Canny(denoised, edge_sens // 2, edge_sens)

    # Д. Объединение и СКЕЛЕТИЗАЦИЯ (схлопывание двойных линий)
    combined = cv2.bitwise_or(edges, shadow_details)
    
    # Математическое утончение до 1 пикселя
    skeleton = np.zeros(combined.shape, np.uint8)
    temp_combined = combined.copy()
    skel_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    # Цикл схлопывания жирных линий в тонкие
    for _ in range(3): # 3 итерации достаточно для большинства тату-фото
        eroded = cv2.erode(temp_combined, skel_kernel)
        temp = cv2.dilate(eroded, skel_kernel)
        temp = cv2.subtract(temp_combined, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        temp_combined = eroded.copy()
        if cv2.countNonZero(temp_combined) == 0: break

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[skeleton > 0] = selected_color

    st.image(preview_img, caption="Результат с усилением теней и тонким контуром", use_column_width=True)

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

    pdf_data = create_pdf(skeleton, target_width_cm, selected_color)
    st.sidebar.download_button(label="📥 Скачать PDF", data=pdf_data, file_name="angar_stencil.pdf", mime="application/pdf")
    
