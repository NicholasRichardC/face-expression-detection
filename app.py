import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import pandas as pd
from collections import deque

st.set_page_config(page_title="Face Expression Detection", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model_and_resources():
    model_path = "model.keras"

    if not os.path.exists(model_path):
        return None, None, "❌ File model.keras tidak ditemukan!", None

    try:
        model = tf.keras.models.load_model(model_path)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        return model, face_cascade, "Model Loaded", model.input_shape
    except Exception as e:
        return None, None, str(e), None


model, face_cascade, status_msg, model_input_shape = load_model_and_resources()

class_names = ['Angry 😠', 'Fear 😨', 'Happy 😄', 'Neutral 😐', 'Sad 😢', 'Surprise 😲']

st.title("🧠 Face Expression Detection (Camera Input Version)")

st.sidebar.header("⚙️ Pengaturan")
st.sidebar.success(status_msg)

picture = st.camera_input("📸 Ambil gambar wajah menggunakan kamera")

if picture is not None:
    img = Image.open(picture)
    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        st.warning("Tidak ada wajah terdeteksi!")
    else:
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]

            # Preprocess
            target_h, target_w = model_input_shape[1], model_input_shape[2]

            face_resized = cv2.resize(face_roi, (target_w, target_h))
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=-1)
            face_resized = np.expand_dims(face_resized, axis=0)

            prediction = model.predict(face_resized)[0]
            idx = np.argmax(prediction)
            prob = prediction[idx] * 100

            st.subheader(f"Ekspresi Terdeteksi: **{class_names[idx]}** ({prob:.2f}%)")

            df = pd.DataFrame({
                "Emosi": class_names,
                "Confidence (%)": prediction * 100
            })

            st.bar_chart(df.set_index("Emosi"))

    st.image(img, caption="Gambar yang diambil", use_column_width=True)
