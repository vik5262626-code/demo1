import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------
# STREAMLIT CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Digit Recognition", layout="centered")

HDR_MODEL_PATH = "hdr_cnn.keras"
CRNN_MODEL_PATH = "crnn_svhn_sequence.keras"

# --------------------------------------------------
# CTC LOSS (FOR MODEL LOADING ONLY)
# --------------------------------------------------
def ctc_loss(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_hdr_model():
    return tf.keras.models.load_model(HDR_MODEL_PATH, compile=False)

@st.cache_resource
def load_crnn_model():
    training_model = tf.keras.models.load_model(
        CRNN_MODEL_PATH,
        custom_objects={"ctc_loss": ctc_loss},
        compile=False
    )

    # Inference model (remove CTC Lambda)
    prediction_model = tf.keras.models.Model(
        inputs=training_model.input[0],
        outputs=training_model.get_layer("dense").output
    )
    return prediction_model

hdr_model = load_hdr_model()
crnn_model = load_crnn_model()

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
def preprocess_hdr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    return img.reshape(1, 28, 28, 1)

def preprocess_crnn(img):
    img = cv2.resize(img, (128, 32))
    img = img.astype("float32") / 255.0
    return img.reshape(1, 32, 128, 3)

# --------------------------------------------------
# CTC DECODER
# --------------------------------------------------
def decode_crnn(pred):
    pred = np.argmax(pred, axis=-1)
    results = []
    for seq in pred:
        prev = -1
        chars = []
        for p in seq:
            if p != prev and p != -1:
                chars.append(str(p))
            prev = p
        results.append("".join(chars))
    return results[0]

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("✏️ Digit Recognition App")

model_choice = st.selectbox(
    "Select Model",
    ["Handwritten Digit (CNN)", "Multi-Digit Sequence (CRNN)"]
)

input_method = st.radio(
    "Input Method",
    ["Draw on Canvas", "Upload Image"]
)

image = None

# --------------------------------------------------
# DRAW CANVAS
# --------------------------------------------------
if input_method == "Draw on Canvas":
    canvas = st_canvas(
        fill_color="black",
        stroke_width=12,
        stroke_color="white",
        background_color="black",
        height=200,
        width=200,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas.image_data is not None:
        image = canvas.image_data.astype(np.uint8)

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
else:
    uploaded = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        image = np.array(Image.open(uploaded).convert("RGB"))

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if image is not None:
    st.image(image, caption="Input Image", width=200)

    if st.button("Predict"):
        if model_choice == "Handwritten Digit (CNN)":
            img = preprocess_hdr(image)
            pred = hdr_model.predict(img, verbose=0)
            digit = np.argmax(pred)
            st.success(f"🧠 Predicted Digit: **{digit}**")

        else:
            img = preprocess_crnn(image)
            pred = crnn_model.predict(img, verbose=0)
            text = decode_crnn(pred)
            st.success(f"🧠 Predicted Number: **{text}**")
