import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('💻 Laptop Price Predictor')

company = st.selectbox('Brand', sorted(df['Company'].unique()))
type = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (GB)', sorted(df['Ram'].unique()))
weight = st.number_input('Weight (Kg)', min_value=0.0)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])
screen_size = st.text_input('Screen Size (inches)', value='15.6')
resolution = st.selectbox(
    'Screen Resolution',
    sorted(['1920x1080','1366x768','1600x900','3840x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
)
cpu = st.selectbox('CPU Brand', sorted(df['Cpu Brand'].unique()))
hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048,4096])
ssd = st.selectbox('SSD (GB)', [0,128,256,512,1024])
gpu = st.selectbox('GPU Brand', sorted(df['Gpu Brand'].unique()))
os = st.selectbox('Operating System', sorted(df['OS'].unique()))

if st.button('Predict'):
    try:
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2 + Y_res**2)**0.5) / float(screen_size)

        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)

        predicted_price = int(np.exp(pipe.predict(query)[0]))
        st.success(f"The predicted price is: {predicted_price:,}")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
