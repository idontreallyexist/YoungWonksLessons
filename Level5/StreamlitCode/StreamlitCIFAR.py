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
photo=None
st.set_page_config(layout="wide")

def new_image():
    global predictions
    global photo
    predictions = probability_model.predict(photo)
    if np.argmax(predictions[0])==i:
        st.subheader("Correct")
        photo=np.array([x_train[random.randint(0,100)]/255.0])
        predictions = probability_model.predict(photo)
        st.session_state['photo']=photo
    else:
        st.subheader("Wrong")
    with placeholder:
        st.image(photo,width=100)

@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return (x_train, y_train),(x_test,y_test)

@st.cache_resource
def load_model():
    return models.load_model('C:/Users/charl/Downloads/Level5/MachineLearning/Tensorflow/TensorflowCIFAR.keras')

(x_train, y_train), (x_test, y_test) = load_data()
model=load_model()
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
st.title("CIFAR Guessing Game")

placeholder=st.container(height=132)
if 'photo' not in st.session_state:
    photo=np.array([x_train[random.randint(0,100)]/255.0])
    predictions = probability_model.predict(photo)
    with placeholder:
        st.image(photo,width=100)
    st.session_state['photo']=photo
else:
    photo=st.session_state['photo']

class_names=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
columns=st.columns(5)

for i in range(0,10):
    if columns[i%5].button(class_names[i],use_container_width=True):
        new_image()