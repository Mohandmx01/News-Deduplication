import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from transformers import pipeline
import re
from nltk.corpus import stopwords


# Preprocess and clean text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load Sentence-BERT model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
def get_embeddings(model, texts):
    return model.encode(texts, convert_to_tensor=True)

# Clustering function
def cluster_articles(embeddings):
    cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8, affinity='cosine', linkage='average')
    return cluster_model.fit_predict(embeddings)

# Summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def get_summary(summarizer, text):
    return summarizer(text, max_length=130, min_length=50, do_sample=False)[0]['summary_text']


st.title("News Deduplication Dashboard")
st.write("This dashboard clusters and summarizes similar news articles to minimize redundancy.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with articles", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Sample Data", data.head())

    # Preprocess and embed articles
    model = load_embedding_model()
    summarizer = load_summarizer()

    data['cleaned_text'] = data['text'].apply(preprocess_text)
    embeddings = get_embeddings(model, data['cleaned_text'].tolist())

    # Perform clustering
    data['cluster'] = cluster_articles(embeddings)

    # Display deduplication results
    st.write("### Deduplicated Clusters")
    final_summaries = []
    for cluster_id, cluster_data in data.groupby('cluster'):
        representative_text = cluster_data.loc[cluster_data['text'].str.len().idxmax()]['text']
        summary = get_summary(summarizer, representative_text)
        final_summaries.append({"cluster_id": cluster_id, "summary": summary})

    # Show summarized data
    final_summary_df = pd.DataFrame(final_summaries)
    st.write(final_summary_df)
