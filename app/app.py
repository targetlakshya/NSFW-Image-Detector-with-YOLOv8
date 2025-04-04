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

# Add custom CSS for centering and theming
st.markdown("""
    <style>
        /* Global dark theme */
        .stApp {
            background-color: #0a192f;
            color: white;
            text-align: center;
            font-family: 'Arial', sans-serif;
        }

        /* Glassmorphism effect */
        .glass {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
        }

        /* Heading Styling */
        h1 {
            color: #64ffda;
            font-size: 2.5rem;
            text-shadow: 2px 2px 10px rgba(100, 255, 218, 0.8);
        }

        /* Buttons with hover effect */
        .stButton>button {
            background-color: #112240;
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
            border: 2px solid #64ffda;
        }

        .stButton>button:hover {
            background-color: #64ffda;
            color: #0a192f;
            transform: scale(1.1);
            box-shadow: 0px 0px 15px rgba(100, 255, 218, 0.6);
        }

        /* Animated images */
        img {
            display: block;
            margin: 0 auto;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(100, 255, 218, 0.5);
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        }

        img:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 20px rgba(100, 255, 218, 0.8);
        }

        /* Text Inputs */
        .stTextInput>div>div>input {
            background-color: #1f2a48;
            color: white;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #64ffda;
        }

        /* NSFW Warning */
        .nsfw-warning {
            color: red;
            font-size: 24px;
            font-weight: bold;
            text-shadow: 0px 0px 8px rgba(255, 0, 0, 0.6);
        }

        /* Not NSFW Text */
        .not-nsfw {
            color: green;
            font-size: 24px;
            font-weight: bold;
            text-shadow: 0px 0px 8px rgba(0, 255, 0, 0.6);
        }

        /* Centering Streamlit Components */
        .css-18ni7ap.e8zbici2 {
            justify-content: center;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            h1 {
                font-size: 2rem;
            }
            .stButton>button {
                font-size: 14px;
                padding: 10px 20px;
            }
        }
    </style>
""", unsafe_allow_html=True)
    

def classify_image(image):
    results = classification_model(image, verbose=True)
    category = results[0].probs.top5
    return category

def segment_image(image):
    results = segmentation_model(image, agnostic_nms=True, retina_masks=True, verbose=True)
    return results

def simplify_classification(top_class_index):
    top_class_name = names[top_class_index].lower()
    nsfw_keywords = ["porn", "hentai", "sexy", "nude", "xxx"]
    if top_class_name in nsfw_keywords:
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

        img_is_explicit = False if segmentation_results[0].boxes.cls.cpu().tolist() == [] else True

        if current_image_path in st.session_state.results_cache:
            cached_results = st.session_state.results_cache[current_image_path]
            classification = cached_results["category"]
            st.success(f"**Classification Result:** {classification}")
        else:
            with st.spinner("Classifying image..."):
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(classify_image, image)
                    category = future.result()
                    classification = simplify_classification(category[0])
                    st.session_state.results_cache[current_image_path] = {"category": classification}
                    st.success(f"**Classification Result:** {classification}")

        if classification == "NSFW" or img_is_explicit:
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
    st.warning("Upload images to begin.")

st.markdown("---")
st.markdown("📌 **Tip:** NSFW classification and visual marking happens automatically.")
st.markdown("Made with ❤️ by Lakshya", unsafe_allow_html=True)