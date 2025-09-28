import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, neighbors
import sklearn.model_selection
from sklearn.model_selection import train_test_split
import joblib
from joblib import dump, load
import tensorflow as tf
from keras import models, layers
import os
import cv2
import matplotlib.pyplot as plt
import random
st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    return models.load_model('C:/Users/charl/Downloads/Level5/MachineLearning/Tensorflow/TensorflowCIFAR.keras')

def predict_image(photo):
    predictions = probability_model.predict(photo)
    return predictions

model=load_model()
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
st.title("Image Classifier")
class_names=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
columns=st.columns(2)
tab1,tab2=st.tabs(['Use Camera','Upload Image'])

photo = tab1.camera_input("Take a photo")
if photo:
    tab1.image(photo)
    pilimage = Image.open(photo).convert("RGB")
    photo = np.asarray(pilimage)
    photo = np.array([np.resize(photo,(32,32,3))])
    predictions=predict_image(photo)
    tab1.subheader(class_names[np.argmax(predictions[0])])
photo = tab2.file_uploader("Upload an Image")
if photo:
    pilimage = Image.open(photo).convert("RGB")
    photo = ImageOps.exif_transpose(pilimage)
    tab2.image(photo)
    photo = np.asarray(pilimage)
    photo = np.rot90(np.rot90(photo))
    photo = np.array([np.resize(photo,(32,32,3))])
    predictions=predict_image(photo)
    tab2.subheader(class_names[np.argmax(predictions[0])])