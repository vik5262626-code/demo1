import streamlit as st
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
st.set_page_config("Digit Recognition", layout="centered")
CONF_THRESHOLD = 0.75
BLANK_TOKEN = 10  # For CTC

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_single_digit_model():
    return tf.keras.models.load_model("mnist_cnn.h5")

@st.cache_resource
def load_sequence_model():
    return tf.keras.models.load_model("crnn_svhn_sequence.keras")

single_digit_model = load_single_digit_model()
sequence_model = load_sequence_model()

# =========================================================
# PREPROCESSING
# =========================================================
def preprocess_single(img):
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img) / 255.0
    return arr.reshape(1, 28, 28, 1)

def preprocess_sequence(img):
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((128, 32))
    arr = np.array(img) / 255.0
    return arr.reshape(1, 32, 128, 1)

# =========================================================
# CTC DECODING
# =========================================================
def ctc_decode(preds):
    seq = np.argmax(preds, axis=-1)[0]
    conf = np.max(preds, axis=-1)[0]

    decoded = []
    decoded_conf = []

    prev = -1
    for i, s in enumerate(seq):
        if s != prev and s != BLANK_TOKEN:
            decoded.append(str(s))
            decoded_conf.append(conf[i])
        prev = s

    return "".join(decoded), decoded_conf

# =========================================================
# CONFIDENCE VISUALIZATION
# =========================================================
def plot_digit_confidence(preds):
    df = pd.DataFrame({
        "Digit": list(range(10)),
        "Confidence": preds[0]
    })
    st.bar_chart(df.set_index("Digit"))

def plot_sequence_confidence(conf):
    df = pd.DataFrame({
        "Position": range(1, len(conf) + 1),
        "Confidence": conf
    })
    st.bar_chart(df.set_index("Position"))
    st.info(f"Average Confidence: **{np.mean(conf):.2f}**")

# =========================================================
# GRAD-CAM
# =========================================================
def gradcam(model, img, layer_name="conv2d"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img)
        loss = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_out), axis=-1)[0]
    heatmap = np.maximum(heatmap, 0)
    return heatmap / np.max(heatmap)

def show_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (28, 28))
    plt.imshow(img[0, :, :, 0], cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")
    st.pyplot(plt)

# =========================================================
# UI
# =========================================================
st.title("🧠 Digit Recognition System")

task = st.radio("Choose Task", ["Single Digit", "Digit Sequence"])
method = st.radio("Input Method", ["Draw", "Upload"])

# =========================================================
# SINGLE DIGIT
# =========================================================
if task == "Single Digit":
    st.subheader("🔢 Single Digit Recognition")

    if method == "Draw":
        canvas = st_canvas(
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            key="single_draw"
        )

        if st.button("Predict"):
            img = Image.fromarray(canvas.image_data.astype("uint8"))
            x = preprocess_single(img)
            preds = single_digit_model.predict(x)

            digit = np.argmax(preds)
            confidence = np.max(preds)

            st.success(f"Prediction: **{digit}** ({confidence:.2f})")

            if confidence < CONF_THRESHOLD:
                st.warning("⚠️ Low confidence prediction")

            st.subheader("Confidence Distribution")
            plot_digit_confidence(preds)

            st.subheader("Model Attention (Grad-CAM)")
            heatmap = gradcam(single_digit_model, x)
            show_heatmap(x, heatmap)

    else:
        uploaded = st.file_uploader("Upload Image", ["png", "jpg"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, width=150)

            if st.button("Predict"):
                x = preprocess_single(img)
                preds = single_digit_model.predict(x)

                digit = np.argmax(preds)
                confidence = np.max(preds)

                st.success(f"Prediction: **{digit}** ({confidence:.2f})")
                plot_digit_confidence(preds)

# =========================================================
# DIGIT SEQUENCE
# =========================================================
else:
    st.subheader("🔢 Digit Sequence Recognition")

    if method == "Draw":
        canvas = st_canvas(
            stroke_width=8,
            stroke_color="white",
            background_color="black",
            height=150,
            width=500,
            key="seq_draw"
        )

        if st.button("Predict"):
            img = Image.fromarray(canvas.image_data.astype("uint8"))
            x = preprocess_sequence(img)
            preds = sequence_model.predict(x)

            text, conf = ctc_decode(preds)

            st.success(f"Predicted Sequence: **{text}**")
            plot_sequence_confidence(conf)

            if np.mean(conf) < CONF_THRESHOLD:
                st.warning("⚠️ Low confidence sequence")

            st.table(pd.DataFrame({
                "Digit": list(text),
                "Confidence": conf
            }))

    else:
        uploaded = st.file_uploader("Upload Image", ["png", "jpg"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, width=300)

            if st.button("Predict"):
                x = preprocess_sequence(img)
                preds = sequence_model.predict(x)

                text, conf = ctc_decode(preds)

                st.success(f"Predicted Sequence: **{text}**")
                plot_sequence_confidence(conf)
