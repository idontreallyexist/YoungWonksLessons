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

@keras.saving.register_keras_serializable(package='my_package')
class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units, **kwargs):
    super().__init__()
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.rnn_units = rnn_units

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(tf.shape(x)[0])
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x
  
  def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "rnn_units": self.rnn_units,
        })
        return config

  @classmethod
  def from_config(cls,config):
    return cls(**config)

@keras.saving.register_keras_serializable(package='my_package2')
class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0, **kwargs):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

  def get_config(self):
        config = super().get_config()
        config.update({
            "model": self.model,
            "chars_from_ids": self.chars_from_ids,
            "ids_from_chars": self.ids_from_chars,
            "temperature": self.temperature,
        })
        return config

  @classmethod
  def from_config(cls, config):
        model = keras.saving.deserialize_keras_object(config.pop("model"))
        chars_from_ids = keras.saving.deserialize_keras_object(config.pop("chars_from_ids"))
        ids_from_chars = keras.saving.deserialize_keras_object(config.pop("ids_from_chars"))
        return cls(model=model,
                   chars_from_ids=chars_from_ids,
                   ids_from_chars=ids_from_chars,
                   **config)

@st.cache_resource
def load_model():
    return models.load_model('C:/Users/charl/Downloads/Github/YoungWonksLessons/Level5/MachineLearning/Tensorflow/News.keras')

df = tfds.load('ag_news_subset', split='train', shuffle_files=True, download=False)
text=tfds.as_dataframe(df.take(10000))['title'].to_string()
print(f'Length of text: {len(text)} characters')
vocab = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
model=load_model()
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

text_prompt=st.text_input("Type Prompt Here")

if st.button("Generate"):
    states = None
    next_char = tf.constant([text_prompt])
    result = [next_char]

    for n in range(100):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    text=tf.strings.join(result)[0].numpy().decode("utf-8")
    st.write(text)
