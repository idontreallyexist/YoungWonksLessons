import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump, load
import tensorflow as tf
from keras import models, layers
import keras
import tensorflow_datasets as tfds

model=tf.load("LyricGen.keras")

maxlen = 40
step = 3
sentences = []
next_chars = []

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

df=pd.read_csv("song_lyrics.csv")[["title","tag","artist","views","lyrics"]].iloc[::10000]
text=df["lyrics"].str.cat(sep=" ")
text = text.replace("\n", " ")
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

text_prompt=st.text_input("Type Prompt Here")
if st.button("Generate"):
    sentence = text_prompt
    for i in range(50):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, 0.2)
        next_char = indices_char[next_index]
        sentence = sentence[1:] + next_char
        generated += next_char
