import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.applications import DenseNet121

# ===============================
# Load models
# ===============================
svm_model = joblib.load("svm_model_densenet121.pkl")
scaler = joblib.load("scaler_densenet121.pkl")

# ===============================
# DenseNet feature extractor
# ===============================
base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

feature_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

base_model.trainable = False

IMG_SIZE = (224,224)

# ===============================
# UI
# ===============================
st.title("🤖 Anemia Detection (SVM + DenseNet Features)")

uploaded_file = st.file_uploader("Upload Palm Image", type=["jpg","png","jpeg"])

# ===============================
# Prediction Function
# ===============================
def predict_svm(img):
    # 🔥 FIX: Convert to RGB
    img = img.convert("RGB")

    img = img.resize(IMG_SIZE)
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    features = feature_model.predict(img_array, verbose=0)
    features = features.flatten().reshape(1, -1)

    features_scaled = scaler.transform(features)

    pred = svm_model.predict(features_scaled)[0]
    prob = svm_model.predict_proba(features_scaled)[0]

    if pred == 0:
        return "Anemic", prob[0]
    else:
        return "Non-Anemic", prob[1]

# ===============================
# Run App
# ===============================
if uploaded_file is not None:
    # 🔥 Convert here also (extra safe)
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, conf = predict_svm(img)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {conf*100:.2f}%")