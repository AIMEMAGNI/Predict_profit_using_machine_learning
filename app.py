import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import joblib
import sklearn



# st.info(" For proper visualisations of this app, please change the theme to dark, by clicking the menu icon in upright corner> settings > Theme>Dark.")

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
    st.markdown(f'<h1 style="color:#FFFFFF;font-size:42px;">{"Company Profit prediction using Machine Learning"}</h1>', unsafe_allow_html=True)     
    st.image("image.png",use_column_width='always')




    
if selected == "Project":
    st.markdown(f'<h1 style="color:#FFFFFF;font-size:40px;">{"Problem background information"}</h1>', unsafe_allow_html=True)     

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
    
    st.markdown(f'<h1 style="color:#FFFFFF;font-size:40px;">{"Let us explore the solution."}</h1>', unsafe_allow_html=True)     

    

    
    
    st.subheader(f"1. Data")
    df = pd.read_csv("online.csv")
    
    st.success('This dataset only shows the expenditures and profit from an unrevealed company. It does not show their investiments or capital. However it was used to showcase how Machine Learning model would help in profit prediction, because Machine Learning model uses the similar approach for prediction.')
    st.dataframe(df, 1200, 400)
    
    st.write("Click the link below to view the original source of dataset.")
    st.write("https://www.kaggle.com/datasets/rahuljoysoyal/onlinecsv")
    
    st.subheader(f"2. Model")
    st.write("In this project we used Logistic Regression model.")
    
    st.subheader(f"3. Model accuracy")
    st.write("The overall accuracy of the model is "+ str(float(0.8982240277293291)))
    
    
    

 
if selected == "Contact":
    st.markdown(f'<h1 style="color:#FFFFFF;font-size:42px;">{"Contact us"}</h1>', unsafe_allow_html=True)     

    
    # from st_functions import st_button, load_css
    import webbrowser
    
    icon_size = 25
    
#     st.button('linkedin', icon_size)
#     st.button('twitter', icon_size)
#     st.button('youtube', icon_size)
    
    st.markdown(f'''<a href={'https://www.linkedin.com/in/aime-magnifique-ndayishimiye-037594213/'}>
    <button style="background-color:Green;">Linkedin</button></a>''', unsafe_allow_html=True)
    
    st.markdown(f'''<a href={'https://twitter.com/aime_magnifique'}>
    <button style="background-color:Green;">Twitter</button></a>''', unsafe_allow_html=True)
    
    st.markdown(f'''<a href={'https://www.youtube.com/'}>
    <button style="background-color:Green;">Youtube</button></a>''', unsafe_allow_html=True)
    
#     if st.button('linkedin'):
#         webbrowser.open_new('https://www.linkedin.com/in/aime-magnifique-ndayishimiye-037594213/')
#     elif st.button('twitter'):
#         webbrowser.open_new_tab('https://twitter.com/aime_magnifique')
        
#     elif st.button('youtube'):
#         webbrowser.open_new_tab('https://www.youtube.com/')
    
    # st.write("To view this project on github, click the link below.")
    # st.write("https://github.com/AIMEMAGNI/Predict_profit_using_machine_learning")
    
    
   

if selected == "Demo":
    
    st.success('This demo uses the model only trained on dataset showed in Project section, so the predictions might not be too practical.')

    st.markdown(f'<h1 style="color:#FFFFFF;font-size:35px;">{"Test our model here"}</h1>', unsafe_allow_html=True)     
    
    input1 =st.number_input('Enter the Marketing spend to predict profit', step=1, value=0)
    input2 =st.number_input('Enter admisnistration fees to predict profit',step=1, value=0)
    input3 =st.number_input('Enter transport fees to predict profit',step=1, value=0)

    input_to_predict = [input1, input2, input3]
    


    if input_to_predict is not None:

        submit = st.button('Predict')

        if submit:
            
            m_jlib = joblib.load('./model_jlib')
            prediction = m_jlib.predict([input_to_predict]) 
            st.subheader('Predicted profit is '+str(int(prediction[0])))
            
          
