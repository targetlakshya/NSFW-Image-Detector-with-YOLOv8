import os
import streamlit as st

media_dir_root = "uploaded_media"
image_dir = f'{media_dir_root}/images'
video_dir = f'{media_dir_root}/videos'

"""
    We are guaranteed to have images and videos dir already.
    app.py is executed first which creates the dirs if they do not exist already.
    Duplicate downloads are not a problem as the zip is stored in the buffer so it can be re-downloaded multiple times.
"""

def delete_uploaded_images():
    if os.path.exists(image_dir):
        for root, _, files in os.walk(image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
        return True
    return False

def delete_uploaded_videos():
    if os.path.exists(video_dir):
        for root, _, files in os.walk(video_dir):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
        return True
    return False

def set_page_configs():
    st.set_page_config(
        page_title='NSFW Detector by Lakshya',
        page_icon='',
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None
    )

    st.markdown(
        """
        <script>
        function isPhone() {
            return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        }

        // Set layout to 'centered' if the user is on a phone
        if (isPhone()) {
            document.body.classList.add('phone-layout');
        }
        </script>
        <style>
            @media (prefers-color-scheme: light) {
            .stApp {
                background-color: #ECEFF4;
                color: #2E3440;
                font-family: 'Inter', sans-serif;
            }

            h1, h2, h3, h4, h5, h6 {
                color: #2E3440;
                font-weight: 600;
            }

            .stButton>button {
                background-color: #81A1C1;
                color: #ECEFF4;
                border-radius: 8px;
                border: none;
                padding: 10px 20px;
                font-weight: 500;
                transition: all 0.3s ease;
            }

            .stButton>button:hover {
                background-color: #5E81AC;
                color: #ECEFF4;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .stFileUploader>div>div>button {
                background-color: #E5E9F0;
                color: #2E3440;
                border-radius: 8px;
                border: 1px solid #D8DEE9;
                padding: 8px 12px;
            }

            .stSlider>div>div>div>div {
                background-color: #81A1C1;
                border-radius: 8px;
            }

            .stCheckbox>label {
                color: #2E3440;
                font-weight: 500;
            }

            .stImage>img {
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }

            .stImage>img:hover {
                transform: scale(1.02);
            }

            .stMarkdown {
                margin-bottom: 1.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
