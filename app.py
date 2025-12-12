import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from collections import deque

st.set_page_config(page_title="Face Expression Detection", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .emotion-box {
        font-size: 24px;
        padding: 10px;
        background-color: #f2f2f2;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_resources():
    model_path = "model.keras"

    if not os.path.exists(model_path):
        return None, None, "❌ File model.keras tidak ditemukan!", None

    try:
        model = tf.keras.models.load_model(model_path)
        input_shape = model.input_shape

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        return model, face_cascade, "Model Loaded", input_shape
    except Exception as e:
        return None, None, str(e), None


model, face_cascade, status_msg, model_input_shape = load_model_and_resources()


class_names = ['Angry 😠', 'Fear 😨', 'Happy 😄', 'Neutral 😐', 'Sad 😢', 'Surprise 😲']

prediction_buffer = deque(maxlen=5)

def preprocess_image(face_roi, target_shape):
    target_h, target_w = target_shape[1], target_shape[2]
    target_channels = target_shape[3]

    resized = cv2.resize(face_roi, (target_w, target_h))

    if target_channels == 1:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        resized = np.expand_dims(resized, axis=-1)
    else:
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    resized = resized.astype("float32") / 255.0
    return np.expand_dims(resized, axis=0)


st.sidebar.header("⚙️ Pengaturan")

run_camera = st.sidebar.toggle("Nyalakan Kamera", value=False)

st.sidebar.info("Model digunakan: **model.keras**")
st.sidebar.success(status_msg)

st.title("🧠 Face Expression Detection Real-Time")

col1, col2 = st.columns([2, 1])

FRAME_WINDOW = col1.image([], use_column_width=True)
emotion_box = col2.empty()
chart_box = col2.empty()

if run_camera:

    camera = cv2.VideoCapture(0)

    while run_camera:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)

        final_prediction = None

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            try:
                processed = preprocess_image(face_roi, model_input_shape)
                prediction = model.predict(processed, verbose=0)[0]

                prediction_buffer.append(prediction)
                smooth_pred = np.mean(prediction_buffer, axis=0)

                idx = np.argmax(smooth_pred)
                final_prediction = (class_names[idx], smooth_pred[idx] * 100, smooth_pred)

                # Bounding box
                color = (0, 255, 0) if idx not in [0, 4] else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
    frame, f"{class_names[idx]} {smooth_pred[idx] * 100:.1f}%",
    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
)

            except:
                pass

        if final_prediction:
            label, prob, full_pred = final_prediction

            emotion_box.markdown(
                f"<div class='emotion-box'>Dominan: {label}<br>({prob:.1f}%)</div>",
                unsafe_allow_html=True
            )

            df = pd.DataFrame({
                "Emosi": class_names,
                "Confidence (%)": full_pred * 100
            })
            chart_box.write(df.style.format({"Confidence (%)": "{:.2f}%"}))

        else:
            emotion_box.markdown(
                "<div class='emotion-box'>Menunggu wajah...</div>",
                unsafe_allow_html=True
            )
            chart_box.empty()
    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb, use_column_width=True)

    camera.release()
