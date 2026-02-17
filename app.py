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
    
    st.sidebar.subheader("Точность линий (без жирности)")
    shadow_detail = st.sidebar.slider("Проработка теней", 0, 10, 3)
    noise_reduction = st.sidebar.slider("Чистота", 1, 10, 2)
    edge_sensitivity = st.sidebar.slider("Чувствительность", 10, 250, 120)
    # Сделали шаг регулировки толщины еще более точным
    line_thickness = st.sidebar.slider("Толщина финальной линии", 1, 3, 1)

    # --- УЛУЧШЕННЫЙ АЛГОРИТМ ТОНКИХ ЛИНИЙ ---
    
    # 1. Подготовка (CLAHE приглушен для чистоты)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    denoised = cv2.medianBlur(enhanced_gray, (noise_reduction * 2 - 1) if noise_reduction > 0 else 1)
    smooth = cv2.bilateralFilter(denoised, 7, 50, 50)
    
    # 2. Основной контур
    edges_main = cv2.Canny(smooth, edge_sensitivity // 2, edge_sensitivity)
    
    # 3. Дополнительные линии теней (только если нужно)
    if shadow_detail > 0:
        block_size = 3 + (shadow_detail * 2)
        soft_edges = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, block_size, 2)
        soft_edges = cv2.bitwise_not(soft_edges)
        # Убираем шум из теней, чтобы не было "каши"
        soft_edges = cv2.morphologyEx(soft_edges, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        combined = cv2.bitwise_or(edges_main, soft_edges)
    else:
        combined = edges_main

    # 4. ФИНАЛЬНОЕ УТОНЧЕНИЕ (Скелетизация упрощенно)
    # Это гарантирует, что даже если линии слиплись, они станут тонкими
    if line_thickness == 1:
        # Оставляем только "хребет" линии
        kernel = np.ones((2,2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_HITMISS, kernel) 
        # Если хитмисс слишком агрессивен, вернемся к обычному тонкому Canny
        combined = cv2.Canny(combined, 100, 200) 

    # --- ОТОБРАЖЕНИЕ ---
    h, w = gray.shape
    alpha = bg_opacity / 100.0
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    blended_bg = cv2.addWeighted(img_array, alpha, white_bg, 1 - alpha, 0)
    
    preview_img = blended_bg.copy()
    preview_img[combined > 0] = selected_color

    st.image(preview_img, caption="Результат: Тонкие и четкие линии", use_column_width=True)

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

    pdf_data = create_pdf(combined, target_width_cm, selected_color)
    
    st.sidebar.markdown("---")
    st.sidebar.download_button(label="📥 Скачать PDF", data=pdf_data, file_name="angar_stencil.pdf", mime="application/pdf")
    
