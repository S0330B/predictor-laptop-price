import streamlit as st
import pickle
import numpy as np


pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Laptop Price Predictor')

company = st.selectbox(
    'Brand',
    sorted(df['Company'].unique())
)


type = st.selectbox(
    'Type',df['TypeName'].unique()
)

ram = st.selectbox(
    'Ram(GB)',sorted(df['Ram'].unique())
)

weight = st.number_input(
    'Weight(Kg)'
)

touchscreen = st.selectbox(
    'Touchscreen',
['No', 'Yes']
)

ips = st.selectbox(
    'IPS',['No', 'Yes']
)

screen_size = st.number_input(
    'Screen Size', min_value=1, max_value=100, step=0.1
)

resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x1800','2880x1800','2560x1600','2560x1440','2304x1441'])

cpu = st.selectbox(
    'CPU',
    sorted(df['Cpu Brand'].unique())
)

hdd = st.selectbox(
    'HDD(GB)',
    [0,128,256,512,1024,2048,4096])

ssd = st.selectbox(
    'SSD(GB)',
    [0,128,256,512,1024])

gpu = st.selectbox(
    'GPU',
    sorted(df['Gpu Brand'].unique())
)

os = st.selectbox(
    'OS',
    sorted(df['OS'].unique())
)

if st.button('Predict'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips == 'Yes':
        ips=1
    else:
        ips=0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi=((X_res**2)+(Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))