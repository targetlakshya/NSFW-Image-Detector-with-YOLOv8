import tempfile
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from PIL import Image, ImageFilter
import os
import cv2
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
    category = results[0].probs.top5
    return category

def segment_image(image):
    results = segmentation_model(image, agnostic_nms=True, retina_masks=True, verbose=True)
    return results

def override_cls(cat, seg):
    img_is_explicit = False if seg[0].boxes.cls.cpu().tolist() == [] else True
    cat_override = names[cat[0]]
    if img_is_explicit and not cat[0] in [1, 3]:
        if cat[1] == 1:
            cat_override = names[1]
        else:
            cat_override = names[3]
    return cat_override

st.write("Upload image(s) to classify and segment sensitive regions.")

uploaded_files = st.file_uploader(
    "📁 Choose an image...",
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

        img_is_explicit = False if segmentation_results[0].boxes.cls.cpu().tolist() == [] else True

        if current_image_path in st.session_state.results_cache:
            cached_results = st.session_state.results_cache[current_image_path]
            category = cached_results["category"]
            st.success(f"**Classification Result:** {category}")
        else:
            with st.spinner("Classifying image..."):
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(classify_image, image)
                    category = future.result()
                    future = executor.submit(override_cls, category, segmentation_results)
                    result = future.result()
                    st.success(f"**Classification Result:** {result}")

        if (category == 'porn' or category == 'hentai') or img_is_explicit:

            boxes = segmentation_results[0].boxes.xyxy.cpu().tolist()
            clss = segmentation_results[0].boxes.cls.cpu().tolist()
            confs = segmentation_results[0].boxes.conf.cpu().tolist()

            image_with_blur = image.copy()
            image_with_boxes = image.copy()
            annotator = Annotator(image_with_boxes, line_width=2, example=segmentation_results[0].names)

            for box, cls, conf in zip(boxes, clss, confs):
                class_name = segmentation_results[0].names[int(cls)]
                label = f"{class_name} ({conf:.2f})"
                annotator.box_label(box, color=colors(int(cls), True), label=label)

            blur_ratio = st.slider("Blur Ratio", min_value=0, max_value=100, value=85)

            for box in boxes:
                obj = image_with_blur.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                blur_obj = obj.filter(ImageFilter.GaussianBlur(radius=blur_ratio))
                if blur_obj.mode == 'RGBA':
                    blur_obj = blur_obj.convert('RGB')
                image_with_blur.paste(blur_obj, (int(box[0]), int(box[1])))

            blur_sensitive_regions = st.checkbox("Blur sensitive regions", True)
            _, cent_co, _ = st.columns(3)
            if blur_sensitive_regions:
                with cent_co:
                    st.image(image_with_blur, caption="Image with Blurred Regions", use_container_width=True)
            else:
                with cent_co:
                    st.image(image_with_boxes, caption="Image with Detected Regions", use_container_width=True)
    else:
        st.warning("No images to display.")

st.markdown("---")
st.markdown("📌 **Tip:** Upload one or more images to automatically classify and detect sensitive regions. Adjust the blur using the slider above.")
st.markdown("---")
st.markdown("Made with ❤️ by Lakshya", unsafe_allow_html=True)
