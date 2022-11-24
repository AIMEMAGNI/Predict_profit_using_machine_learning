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
import pickle

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
        background-size: cover

    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('./image1.jpg')  

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
    st.image("./image.png",use_column_width='always')


    
if selected == "Project":
    st.title(f"Problem background information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Introduction")
        st.markdown(f'<h1 style="color:#ffffff;font-size:8px;">{"Profit is an indicator of  whether a company is developing, it shows company growth and can secure financing from a bank, attract investors to fund its operations."}', unsafe_allow_html=True)
        
    with col2:
        st.subheader("Problem")
#         st.write(f'<h1 style="color:#33ff33;font-size:24px;">>{"Traditional data tools are only able to compute current profits of companies."}')
        st.markdown(f'<h1 style="color:#ffffff;font-size:8px;">{"Traditional data tools are only able to compute current profits of companies."}', unsafe_allow_html=True)       

    with col3:
        st.subheader("Solution")
        st.markdown(f'<h1 style="color:#ffffff;font-size:8px;">{"Use of well trained Machine Learning models would provide an efficient solution."}', unsafe_allow_html=True)
    
    st.title(f"Let's explore the solution.")

    st.subheader(f"1. Data")
    



    df = pd.read_csv("online.csv")

    st.dataframe(df, 1200, 400)

    # cols = df.shape[1]
    # X = df.iloc[:,:cols-2].values
    # y = df.iloc[:,cols-1 : cols].values

    # # Splitting the dataset into the Training set and Test set
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    # # MODEL
    # from sklearn.linear_model import LinearRegression
    # regressor = LinearRegression()
    # regressor.fit(X_train, y_train)

    # # Predictions
    # y_pred = regressor.predict(X_test)

    # # StandardScaling
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # sc_x_train = scaler.fit_transform(X_train)

    # #PCA
    # from sklearn.decomposition import PCA
    # transformer = PCA(n_components=1, random_state=0)
    # x_train = transformer.fit_transform(sc_x_train)
    # Y_train = transformer.fit_transform(y_train)

    # #PLOT
    # fig = plt.scatter(x_train, Y_train , color = 'red')
    
    # st.pyplot(fig)









    # profile = ProfileReport(df)
    # st.write(profile)






     
if selected == "Contact":
    st.title(f"Contact us")
    st.image("./OG.jpg", width = 400)


if selected == "Demo":
    st.header("Test our model here")
    input1 =st.number_input('Enter the Marketing spend to predict profit')
    input2 =st.number_input('Enter admisnistration fees to predict profit')
    input3 =st.number_input('Enter transport fees to predict profit')

    input_to_predict = [input1, input2, input3]


    # if input_to_predict is not None:

    #     submit = st.button('Predict')

    #     if submit:
    #         # model = keras.models.load_model('my_h5_model.h5')
    #         # # h5=h5py.File('my_h5_model.h5')
    #         # # model.predict(input_to_predict)
            
    #         model = load_module('my_h5_model.h5')
    #         model.predict(input_to_predict)     
