import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# Load CNN model
# ===============================
model = tf.keras.models.load_model("best_model.h5")

IMG_SIZE = (128, 128)

# ===============================
# UI
# ===============================
st.title("🧠 Anemia Detection (CNN Model)")

uploaded_file = st.file_uploader("Upload Palm Image", type=["jpg", "png", "jpeg"])

# ===============================
# Prediction Function
# ===============================
def predict_cnn(img):
    # 🔥 FIX: Convert to RGB (removes alpha channel)
    img = img.convert("RGB")

    # Resize
    img = img.resize(IMG_SIZE)

    # Normalize
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        return "Non-Anemic", pred
    else:
        return "Anemic", 1 - pred

# ===============================
# Run App
# ===============================
if uploaded_file is not None:
    # 🔥 Convert here also (extra safety)
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, conf = predict_cnn(img)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {conf * 100:.2f}%")