import streamlit as st
st.set_page_config(layout="wide")
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import cv2



st.title("Détection d'accident de Circulation")



upload = st.file_uploader("Chargez l'image", type=['png','jpg','jpeg'])

st.cache()
def charge_model():
    model = load_model('model_accident.h5', compile=False)
    return model

model = charge_model()

c1, c2 = st.columns(2)

if upload:
    img = Image.open(upload)
    img_resize = np.asarray(img)
    img_resize = cv2.resize(img_resize, (224,224))
    img_array = img_to_array(img_resize)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array)
    prediction = model.predict(preprocessed_img)

    normal = round(prediction[0][1], 2) * 100
    accident = 100 - normal

    c1.image(img)
    c2.write(f"# Accident   ..........   {accident:.2f} %")
    c2.write(f"# Normal   ..........   {normal:.2f} %")



