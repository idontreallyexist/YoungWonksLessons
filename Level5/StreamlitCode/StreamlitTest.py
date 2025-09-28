import streamlit as st
import pandas as pd
import numpy as np

st.title("Uber pickups in NYC")

DATA_URL = ("uber-raw-data-sep14.csv")
data=pd.read_csv(DATA_URL,nrows=1000)

st.subheader("Raw Data")
st.write(data)

data.columns=data.columns.str.lower()
print(data.columns)
data['date/time']=pd.to_datetime(data['date/time'])
st.subheader("Number of pickups by hour")
hist_values=np.histogram(data['date/time'].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

hour_to_filter=st.slider('hour',0,23,17)
filtered_data=data[data['date/time'].dt.hour == hour_to_filter]

st.subheader("Map of all pickups at %s:00" % hour_to_filter)
st.map(filtered_data)