import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Digit Recognition App", layout="centered")

HDR_MODEL_PATH = "hdr_cnn.keras"
CRNN_MODEL_PATH = "crnn_svhn_sequence.keras"

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_hdr_model():
    return tf.keras.models.load_model(HDR_MODEL_PATH)

@st.cache_resource
def load_crnn_model():
    return tf.keras.models.load_model(CRNN_MODEL_PATH)

hdr_model = load_hdr_model()
crnn_model = load_crnn_model()

# ---------------- PREPROCESSING ----------------
def preprocess_hdr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def preprocess_crnn(img):
    img = cv2.resize(img, (128, 32))
    img = img / 255.0
    img = img.reshape(1, 32, 128, 3)
    return img

def decode_crnn(pred):
    pred = np.argmax(pred, axis=-1)[0]
    digits = [str(d) for d in pred if d != -1]
    return "".join(digits)

# ---------------- UI ----------------
st.title("✏️ Digit Recognition App")

model_choice = st.selectbox(
    "Choose Model",
    ("Handwritten Digit (CNN)", "Sequence Digits (CRNN)")
)

input_method = st.radio(
    "Choose Input Method",
    ("Draw on Canvas", "Upload Image")
)

image = None

# ---------------- DRAW CANVAS ----------------
if input_method == "Draw on Canvas":
    canvas = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas.image_data is not None:
        image = canvas.image_data.astype(np.uint8)

# ---------------- UPLOAD IMAGE ----------------
else:
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        image = np.array(pil_img)

# ---------------- PREDICTION ----------------
if image is not None:
    st.image(image, caption="Input Image", width=200)

    if st.button("Predict"):
        if model_choice == "Handwritten Digit (CNN)":
            processed = preprocess_hdr(image)
            prediction = hdr_model.predict(processed)
            digit = np.argmax(prediction)
            st.success(f"🧠 Predicted Digit: **{digit}**")

        else:
            processed = preprocess_crnn(image)
            prediction = crnn_model.predict(processed)
            sequence = decode_crnn(prediction)
            st.success(f"🧠 Predicted Number Sequence: **{sequence}**")
