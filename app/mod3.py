from setfit import AbsaModel
import pandas as pd
import logic
from tqdm import tqdm
import streamlit as st

model = AbsaModel.from_pretrained(
        "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
        "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
)

def get_keywords():
    try:
        data = pd.read_csv('result_df.csv')
        filtered_data = data[(data["Taste"] == "negative") | (data["Shopping Experience"] == "negative") | (data["Brand Satisfaction"] == "negative")]

        tqdm.pandas()
        filtered_data["Analysis"] = filtered_data["Review"].progress_apply(analyze_review)

    except FileNotFoundError:
        st.write("Please navigate to Statistical Analysis and upload your dataset first.")

def analyze_review(review):
    
    preds = model(review)
    return preds



