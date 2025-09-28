import pandas as pd
import numpy as np
import math
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont
dataDivide=100

st.title("Lunar Terrain Visualizer")
df=pd.read_csv("Nobile1.csv")

#Plot
st.subheader("Height Visualizer")
st.scatter_chart(data=df,x='X',y='Y',x_label="X (meters)",y_label="Y (meters)",color='Height (meters)')
st.subheader("Slope Visualizer")
st.scatter_chart(data=df,x='X',y='Y',x_label="X (meters)",y_label="Y (meters)",color='Slope (degrees)')

#Find lowest slope areas
df.sort_values(by="Slope (degrees)",ascending=False)
dflat=df.head(2)['Latitude'].to_numpy()
dflong=df.head(2)['Longitude'].to_numpy()
dfx=df.head(2)['X'].to_numpy()
dfy=df.head(2)['Y'].to_numpy()

st.text("Optimal Landing Spot (Lowest Slope)")
st.text("("+str(dfx[0])+", "+str(dfy[0])+") or ("+str(dflat[0])+" N, "+str(dflong[0])+" E)")