
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model = load_model("ecg_mi_classifier.h5")
IMG_SIZE = 224

st.set_page_config(page_title="ECG MI Classifier", layout="wide")
st.title(" ECG Myocardial Infarction Detection")
st.markdown("Upload a scanned ECG image to classify as **Normal** or **Myocardial Infarction (MI)**.")

uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        st.image(image, caption="Uploaded ECG Image", use_column_width=True)
        
        # Preprocess image
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img)
        class_names = ['Normal', 'MI']
        prediction = class_names[np.argmax(pred)]

        st.markdown(f"## Prediction: `{prediction}`")
        st.markdown(f"### Confidence: `{pred[0][np.argmax(pred)]*100:.2f}%`")
    else:
        st.error("Unable to read image.")