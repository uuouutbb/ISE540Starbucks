import streamlit as st
import base64

def setbg():
    background_style = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url('https://github.com/uuouutbb/ISE540Starbucks/blob/main/starbuckscup.png?raw=true');
        background-size: 28%;
        background-repeat: no-repeat;
        background-position: bottom right;
        background-color: #006241;
    }
    [data-testid="stSidebar"] {
        background-color: #562F1E;
    }
    [data-testid="stHeader"] {
        background-color: #003220;
        color: white;
    }
    header .css-1lsmgbg.egzxvld1 {
        visibility: hidden; 
    }
    # body {
    #     background: url('https://i.pinimg.com/originals/57/e0/fd/57e0fd6a7e0bde13e3c08b8b06bb831d.jpg');
    #     background-size: cover;
    # }

    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)
    