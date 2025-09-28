import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont

st.title("Selfie")
photo = st.camera_input("Smile for the camera!")
img = Image.open(photo)
flipped = ImageOps.mirror(img)
gray = ImageOps.grayscale(img)
invert = ImageOps.invert(img)
col1, col2, col3 = st.columns(3)
st.subheader("Original")
st.image(img)
st.subheader("Flipped")
st.image(flipped)
if st.button("Show Grayscale"):
    st.subheader("Grayscale")
    st.image(gray)
if st.button("Show Inverted"):
    st.subheader("Inverted")
    st.image(invert)
                    