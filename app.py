import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Customer Feedback Analyzer")

# --- Helper Functions (with caching for performance) ---
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    df.dropna(subset=['Review Text'], inplace=True)
    df.rename(columns={'Review Text': 'text'}, inplace=True)
    return df

@st.cache_resource
def download_stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

def preprocess_text(text, stop_words):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# --- Main Application Logic ---
st.title("👕 Customer Feedback Analysis Engine")

# 1. Load Data
df_full = load_data('reviews.csv')
df_sample = df_full.head(500).copy() # Work with a sample for speed

# 2. Preprocess Text
stop_words = download_stopwords()
df_sample['cleaned_text'] = df_sample['text'].apply(lambda text: preprocess_text(text, stop_words))

# 3. Perform Sentiment Analysis
sentiment_model = load_sentiment_model()
results = df_sample['cleaned_text'].apply(lambda x: sentiment_model(x[:512]))
df_sample['sentiment_label'] = results.apply(lambda res: res[0]['label'])
df_sample['sentiment_score'] = results.apply(lambda res: res[0]['score'])

# --- Sidebar Filters ---
st.sidebar.header("Filter Reviews")
sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    options=df_sample['sentiment_label'].unique(),
    default=df_sample['sentiment_label'].unique()
)

# Apply filter to the dataframe
df_filtered = df_sample[df_sample['sentiment_label'].isin(sentiment_filter)]

# --- Dashboard Display ---
st.header("Dashboard")

if df_filtered.empty:
    st.warning("No data to display for the selected filters.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_filtered['sentiment_label'].value_counts()
        fig = px.pie(
            sentiment_counts,
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Distribution of Sentiments",
            color=sentiment_counts.index,
            color_discrete_map={'POSITIVE': 'lightgreen', 'NEGATIVE': 'salmon'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Key Metrics")
        total_reviews = len(df_filtered)
        positive_reviews = len(df_filtered[df_filtered['sentiment_label'] == 'POSITIVE'])
        negative_reviews = len(df_filtered[df_filtered['sentiment_label'] == 'NEGATIVE'])
        
        st.metric("Total Reviews", f"{total_reviews}")
        st.metric("Positive Reviews", f"{positive_reviews}")
        st.metric("Negative Reviews", f"{negative_reviews}")

    # Display filtered reviews in a table
    st.subheader("Filtered Reviews")
    st.dataframe(df_filtered[['text', 'sentiment_label', 'sentiment_score']])