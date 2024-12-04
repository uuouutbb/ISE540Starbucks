import streamlit as st
import pandas as pd
import numpy as np
import vis
import csv
import UI
import mod3
import os
import seaborn as sns
import sqlite3
from streamlit_option_menu import option_menu
import altair as alt
import pickle
import plotly.express as px
import pydeck as pdk
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('/Users/agtang/Library/Mobile Documents/com~apple~CloudDocs/Documents/USC/540/project/Starbucks_review_sentiment_app/fine_tuned_models22')
model = AutoModelForSequenceClassification.from_pretrained('/Users/agtang/Library/Mobile Documents/com~apple~CloudDocs/Documents/USC/540/project/Starbucks_review_sentiment_app/fine_tuned_models22')

def load_dataset():
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return None
            
            st.success("File uploaded successfully!")
            return df
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
            return None
    else:
        return None

@st.cache_data(show_spinner=False)
def generate_aspect_level(data):
    # tokenizer = AutoTokenizer.from_pretrained('/Users/agtang/Library/Mobile Documents/com~apple~CloudDocs/Documents/USC/540/project/Starbucks_review_sentiment_app/fine_tuned_model2')
    # model = AutoModelForSequenceClassification.from_pretrained('/Users/agtang/Library/Mobile Documents/com~apple~CloudDocs/Documents/USC/540/project/Starbucks_review_sentiment_app/fine_tuned_model2')

    reviews = data['Review']
    aspects = ['Taste', 'Shopping Experience', 'Brand Satisfaction']
    results = {'Review': reviews}
    batch_size = 16

    for aspect in aspects:
        aspect_predictions = []

        for i in range(0, len(reviews), batch_size):
            batch_reviews = reviews[i:i+batch_size].tolist()
            input_texts = [f"{review} [aspect: {aspect}]" for review in batch_reviews]
            inputs = tokenizer(input_texts, truncation=True, padding=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1).tolist()
            reverse_sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive', 3: 'not mentioned'}
            aspect_predictions.extend([reverse_sentiment_map[pred] for pred in predictions])

        results[aspect] = aspect_predictions

    result_df = pd.DataFrame(results)
    result_df.to_csv("result_df.csv", index=False)
    st.dataframe(result_df)
    return result_df

def stat():
    st.header("Upload Your Dataset for Aspect-Based Sentiment Analysis")
    st.write("""Upload a dataset containing reviews (in CSV or Excel format) to analyze sentiments for specific aspects using our fine-tuned large language model. 
             The model will generate labels for each review, categorizing sentiments as positive, negative, neutral, or not mentioned for each aspect.""")
    data_filled = None
    data = load_dataset()
    if data is not None:
        with st.spinner("Processing reviews... Please wait."):
            data_filled = generate_aspect_level(data)
            st.success("Aspect sentiment generation completed!")
    return data_filled

def play_around():
    st.title("Try it")
    st.write("Enter a review:")
    st.markdown("_e.g., I'm obsessed with the Pumpkin Spice Latte—it's like fall in a cup! I love Starbucks!_")
    user_input = st.text_input("")

    if user_input:
        aspects = ['Taste', 'Shopping Experience', 'Brand Satisfaction']
        results = {'Review': user_input}
        reverse_sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive', 3: 'not mentioned'}

        for aspect in aspects:
            input_text = user_input + f" [aspect: {aspect}]"
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1).tolist()
            sentiment = reverse_sentiment_map[predictions[0]]
            results[aspect] = sentiment
            st.write(f"**Aspect:** {aspect}")
            st.write(f"- **Sentiment:** {sentiment}")
            st.write("---")
    

def initialize():

    with st.sidebar:
        selected = option_menu(
        menu_title = "Main Menu",
        options = ["Home","Statistical Analysis","Play around"],
        icons = ["house","graph-up-arrow","book"],
        menu_icon = "cast",
        default_index = 0,
    )
    if selected == "Home":
        home_page()
    elif selected == "Statistical Analysis":
        data_filled = stat()
        if data_filled is not None:
            vis.statistic_charts(data_filled, 3)
            vis.statistic_charts(data_filled, 2)
    elif selected == "Play around":
        play_around()

        

def home_page():
    st.title("Starbucks Coffee Sentiment Review Analyzer")
    st.subheader("Welcome to the Starbucks Coffee Sentiment Review Analyzer!")
    st.write("""
        This app is designed to analyze customer reviews of Starbucks products. 
        It uses a fine-tuned LLM to identify key aspects and sentiments, helping you gain valuable insights into customer feedback.
    """)







