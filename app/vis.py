import streamlit as st
import pandas as pd
import numpy as np
import csv
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


def statistic_charts(data, plot_t):
    long_data = data.melt(
        id_vars=['Review'], 
        var_name='Aspect', 
        value_name='Level'
    )

    level_mapping = {
        'negative': 'Negative',
        'positive': 'Positive',
        'neutral': 'Neutral',
        'not mentioned': 'Not Mentioned'
    }

    match plot_t:
        case 1:
            positive_data = data[data['Level'] == 1]
            aspect_counts = positive_data['Aspect'].value_counts().reset_index()
            aspect_counts.columns = ['Aspect', 'Count']
            aspect_counts['Percentage'] = (aspect_counts['Count'] / aspect_counts['Count'].sum()) * 100

            chart = alt.Chart(aspect_counts).mark_arc().encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Aspect", type="nominal"),
                tooltip=[
                    alt.Tooltip("Aspect", title="Aspect"),
                    alt.Tooltip("Count", title="Count"),
                    alt.Tooltip("Percentage", format=".1f", title="Percentage (%)")
                ]
            ).properties(
                title="Distribution of Positive Levels by Aspect"
            )
            st.altair_chart(chart, use_container_width=True)
            
        case 2:
            filtered_data = long_data[long_data['Level'] != "not mentioned"]
            level_counts = filtered_data.groupby(['Aspect', 'Level']).size().reset_index(name='Count')
            level_counts['Level'] = level_counts['Level'].map(level_mapping)

            st.title("Aspect-Based Sentiment Analysis - Bar Chart")

            fig = px.bar(
                level_counts,
                x='Aspect',
                y='Count',
                color='Level',
                barmode='group',
                title="Sentiment Distribution by Aspect (Excluding 'Not Mentioned')",
                labels={'Count': 'Number of Reviews', 'Aspect': 'Aspect'}
            )

            fig.update_layout(
                xaxis_title="Aspect",
                yaxis_title="Number of Reviews",
                legend_title="Sentiment"
            )
            st.plotly_chart(fig)
        case 3:
            level_counts = long_data.groupby(['Aspect', 'Level']).size().reset_index(name='Count')
            level_counts['Level'] = level_counts['Level'].map(level_mapping)
            st.title("Aspect-Based Sentiment Analysis - Pie Chart for Each Aspect")
            aspect_selected = st.selectbox("Select an Aspect", level_counts['Aspect'].unique())
            filtered_data = level_counts[level_counts['Aspect'] == aspect_selected]
            fig = px.pie(
                filtered_data,
                values='Count',
                names='Level',
                title=f"Distribution of Sentiment Levels for '{aspect_selected}'",
                hole=0.3
            )

            fig.update_traces(textinfo='percent+label', hoverinfo='label+value+percent')
            st.plotly_chart(fig)
        case 4:
            level_counts = data.groupby(['Aspect', 'Level']).size().reset_index(name='Count')

            level_mapping = {
                0: 'Negative',
                1: 'Positive',
                2: 'Neutral',
                3: 'Not Mentioned'
            }
            level_counts['Level'] = level_counts['Level'].map(level_mapping)

            filtered_counts = level_counts[level_counts['Level'] != 'Not Mentioned']

            aspect_selected = st.selectbox("Select an Aspect", filtered_counts['Aspect'].unique())

            aspect_data = filtered_counts[filtered_counts['Aspect'] == aspect_selected]

            fig = px.pie(
                aspect_data,
                values='Count',
                names='Level',
                title=f"Distribution of Levels for Aspect '{aspect_selected}'",
                hole=0.3
            )

            fig.update_traces(textinfo='percent+label', hoverinfo='label+value+percent')

            st.plotly_chart(fig)
        case _:
            return "Default case"
        
def chart(data):
    aspect_selected = st.selectbox("Select an Aspect", level_counts['Aspect'].unique())
    long_data = data.melt(
        id_vars=['Review'], 
        var_name='Aspect', 
        value_name='Level'
    )

    level_counts = long_data.groupby(['Aspect', 'Level']).size().reset_index(name='Count')

    level_mapping = {
        'negative': 'Negative',
        'positive': 'Positive',
        'neutral': 'Neutral',
        'not mentioned': 'Not Mentioned'
    }
    level_counts['Level'] = level_counts['Level'].map(level_mapping)

    st.title("Aspect-Based Sentiment Analysis - Pie Chart for Each Aspect")

    filtered_data = level_counts[level_counts['Aspect'] == aspect_selected]

    fig = px.pie(
        filtered_data,
        values='Count',
        names='Level',
        title=f"Distribution of Sentiment Levels for '{aspect_selected}'",
        hole=0.3
    )

    fig.update_traces(textinfo='percent+label', hoverinfo='label+value+percent')
    
    st.plotly_chart(fig)