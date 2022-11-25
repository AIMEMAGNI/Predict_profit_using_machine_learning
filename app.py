# from imp import load_module
import streamlit as st
from streamlit_option_menu import option_menu
# from load_image import load_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from PIL import Image
import base64
# from pandas_profiling import ProfileReport
# from tensorflow import keras
# import h5py
# import pickle
import joblib
# import cv2
# from tensorflow.keras.models import load_model

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-color: #FFFFFF

    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('image1.jpg')  

def streamlit_menu():
    selected = option_menu(
        menu_title=None, 
        options=["Home", "Project","Demo","Contact"],  
        icons=["house", "book","search", "envelope"], 
        menu_icon="cast",  
        default_index=0,  
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {
                   "font-size": "20px",
                   "color": "black",
                   "text-align": "left",
                   "margin": "0px",
                   "--hover-color": "#eee",
                   },
                   "nav-link-selected": {"background-color": "green"},
                   },
                   ) 
    return selected


selected = streamlit_menu()

if selected == "Home":
    st.title(f"Company Profit prediction using Machine Learning")
    st.image("image.png",use_column_width='always')


    
if selected == "Project":
    st.title(f"Problem background information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Introduction")
        st.write('Profit is an indicator of  whether a company is developing, it shows company growth and can secure financing from a bank, attract investors to fund its operations.')
        
    with col2:
        st.subheader("Problem")
        st.write('Traditional data tools are only able to compute current profits of companies.')

    with col3:
        st.subheader("Solution")
        st.write("Use of well trained Machine Learning models would provide an efficient solution.")
    
    st.title(f"Let's explore the solution.")

    st.subheader(f"1. Data")
    



    df = pd.read_csv("online.csv")

    st.dataframe(df, 1200, 400)
    
    st.write("Click the link below to view the original source of dataset.")
    st.write("https://www.kaggle.com/datasets/rahuljoysoyal/onlinecsv")

 
if selected == "Contact":
    st.title(f"Contact us")
    
    from st_functions import st_button, load_css
    
    icon_size = 25
    
    st_button('linkedin', 'https://www.linkedin.com/in/aime-magnifique-ndayishimiye-037594213/', 'Follow me on LinkedIn', icon_size)
    st_button('twitter', 'https://twitter.com/aime_magnifique', 'Follow me on Twitter', icon_size)
    st_button('youtube', 'https://www.youtube.com/', 'YouTube', icon_size)
    
    st.write("To view this project on github, click the link below.")
    st.write("https://github.com/AIMEMAGNI/Predict_profit_using_machine_learning")
   

if selected == "Demo":
    st.header("Test our model here")
    input1 =st.number_input('Enter the Marketing spend to predict profit', step=1, value=1)
    input2 =st.number_input('Enter admisnistration fees to predict profit',step=1, value=1)
    input3 =st.number_input('Enter transport fees to predict profit',step=1, value=1)

    input_to_predict = [input1, input2, input3]


    if input_to_predict is not None:

        submit = st.button('Predict')

        if submit:
            
            m_jlib = joblib.load('model_jlib')
            prediction = m_jlib.predict([input_to_predict]) 
            st.subheader('Predicted profit is '+str(int(prediction[0])))
            
            

