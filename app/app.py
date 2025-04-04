import tempfile
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image
import os
import cv2
import numpy as np
from pillow_heif import register_heif_opener
import utils
from concurrent.futures import ThreadPoolExecutor

# Enable support for HEIC images
register_heif_opener()

# Set theme and page layout
utils.set_page_configs()

# Directory to save uploaded images
media_dir_root = "uploaded_media"
image_dir = f'{media_dir_root}/images'
os.makedirs(image_dir, exist_ok=True)

@st.cache_resource(ttl=24*3600)
def load_models():
    classification_model = YOLO('models/classification_model.pt')
    segmentation_model = YOLO('models/segmentation_model.pt')
    return classification_model, segmentation_model

classification_model, segmentation_model = load_models()

if "image_index" not in st.session_state:
    st.session_state.image_index = 0

if "saved_image_paths" not in st.session_state:
    st.session_state.saved_image_paths = []

if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}

names = classification_model.names

def classify_image(image):
    results = classification_model(image, verbose=True)
    top_class_index = results[0].probs.top1
    return top_class_index

def segment_image(image):
    results = segmentation_model(image, agnostic_nms=True, retina_masks=True, verbose=True)
    return results

def simplify_classification(top_class_index):
    """Mark images as NSFW only if classified as 'porn'. Everything else is Not NSFW."""
    top_class_name = names[top_class_index].lower()
    if top_class_name == "porn":
        return "NSFW"
    return "Not NSFW"

st.title("🔍 NSFW Image Detector")
st.write("Upload image(s) to classify them as NSFW or Not NSFW.")

uploaded_files = st.file_uploader(
    "📁 Choose image(s)...",
    type=["bmp", "dng", "jpg", "jpeg", "mpo", "png", "tif", "tiff", "webp", "pfm", "HEIC"],
    accept_multiple_files=True
)

if uploaded_files:
    current_uploaded_files = {file.name for file in uploaded_files}
    st.session_state.saved_image_paths = [
        path for path in st.session_state.saved_image_paths
        if os.path.basename(path) in current_uploaded_files
    ]

    for uploaded_file in uploaded_files:
        file_path = os.path.join(image_dir, uploaded_file.name)
        if file_path not in st.session_state.saved_image_paths:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.saved_image_paths.append(file_path)

    if st.session_state.image_index >= len(st.session_state.saved_image_paths):
        st.session_state.image_index = max(0, len(st.session_state.saved_image_paths) - 1)

    if st.session_state.saved_image_paths:
        current_image_path = st.session_state.saved_image_paths[st.session_state.image_index]
        image = Image.open(current_image_path)
        
        _, cent_co, _ = st.columns(3)
        with cent_co:
            st.image(image, caption=f"Image {st.session_state.image_index + 1} of {len(st.session_state.saved_image_paths)}", use_container_width=True)

        col1, _, col3 = st.columns([1, 10, 1])
        with col1:
            if st.button("Previous") and st.session_state.image_index > 0:
                st.session_state.image_index -= 1
                st.rerun()
        with col3:
            if st.button("Next") and st.session_state.image_index < len(st.session_state.saved_image_paths) - 1:
                st.session_state.image_index += 1
                st.rerun()

        segmentation_results = []
        with st.spinner("Detecting sensitive regions..."):
            with ThreadPoolExecutor() as executor:
                future = executor.submit(segment_image, image)
                segmentation_results = future.result()

        # Segmentation only detects explicit regions, doesn't decide NSFW status
        img_is_explicit = len(segmentation_results[0].boxes.cls.cpu().tolist()) > 0

        if current_image_path in st.session_state.results_cache:
            cached_results = st.session_state.results_cache[current_image_path]
            classification = cached_results["category"]
            st.success(f"**Classification Result:** {classification}")
        else:
            with st.spinner("Classifying image..."):
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(classify_image, image)
                    top_class_index = future.result()
                    classification = simplify_classification(top_class_index)
                    st.session_state.results_cache[current_image_path] = {"category": classification}
                    st.success(f"**Classification Result:** {classification}")

        # Show NSFW warning if classified as NSFW
        if classification == "NSFW":
            st.markdown("<p class='nsfw-warning'>⚠️ NSFW Content Detected!</p>", unsafe_allow_html=True)
            
            # Mark detected regions
            boxes = segmentation_results[0].boxes.xyxy.cpu().tolist()
            image_with_circles = np.array(image)
            image_with_circles = cv2.cvtColor(image_with_circles, cv2.COLOR_RGB2BGR)

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                radius = max((x2 - x1), (y2 - y1)) // 3
                cv2.circle(image_with_circles, (center_x, center_y), radius, (0, 0, 255), thickness=4)

            image_with_circles = cv2.cvtColor(image_with_circles, cv2.COLOR_BGR2RGB)
            image_with_circles = Image.fromarray(image_with_circles)

            _, cent_co, _ = st.columns(3)
            with cent_co:
                st.image(image_with_circles, caption="Sensitive regions marked", use_container_width=True)
        else:
            st.markdown("<p class='not-nsfw'>✅ This image is Not NSFW</p>", unsafe_allow_html=True)
else:
    st.warning("Upload images to begin.")

st.markdown("---")
st.markdown("📌 **Tip:** NSFW classification and visual marking happens automatically.")
st.markdown("Made with ❤️ by Lakshya", unsafe_allow_html=True)
