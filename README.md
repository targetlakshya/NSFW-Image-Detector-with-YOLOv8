# NSFW Image Detection with YOLOv8 🚫🖼️

This project is a web-based application built with **Streamlit** that uses **YOLOv8** models for classifying and segmenting explicit content in images. It helps detect NSFW content and automatically blurs sensitive regions.

---

## 🔍 Features

- 📁 Upload and view multiple images
- ✅ Classify images using a custom-trained NSFW classification model
- 🖍️ Segment and blur explicit regions with high accuracy using YOLOv8 segmentation
- 🎛️ Adjustable blur intensity via slider
- 🧠 Caching to avoid reprocessing images
- 💡 Simple, clean UI powered by Streamlit

---

## 🛠️ Installation

```bash
git clone https://github.com/targetlakshya/NSFW-Image-Detector-with-YOLOv8
cd NSFW-Image-Detector-with-YOLOv8
pip install -r requirements.txt

## 📦 Project Structure

.
├── app/                            # Main application folder
│   ├── app.py                      # Streamlit web app
│   ├── utils.py                     # Utility functions
│   ├── __pycache__/                 # Python cache files
│
├── assets/                          # Visual assets
│   ├── classification/              # Classification-related images
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_normalized.png
│   ├── segmentation/                # Segmentation-related images
│       ├── confusion_matrix.png
│       ├── confusion_matrix_normalized.png
│
├── models/                          # Pre-trained YOLOv8 models
│   ├── classification_model.pt
│   ├── segmentation_model.pt
│
├── scripts/                         # Training scripts
│   ├── train_classification.py
│   ├── train_segmentation.py
│
├── uploaded_media/                   # Directory for uploaded images
├── .devcontainer/                     # VS Code DevContainer config
│   ├── devcontainer.json
│
├── venv/                              # Virtual environment (ignored)
├── LICENSE                            # License file
├── .gitignore                         # Git ignore file
├── requirements.txt                    # Python dependencies
└── README.md                          # Project documentation
