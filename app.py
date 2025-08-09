import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from collections import defaultdict
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Keyword Clustering Tool",
    page_icon="🔍",
    layout="wide"
)

# Load the model with caching to avoid reloading on every run
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# App title and description
st.title("AI-Powered Keyword Clustering Tool")
st.markdown("Upload or paste keywords to group them by semantic similarity")

# Input section
input_option = st.radio("Select input method", ["Upload CSV", "Paste keywords"])

if input_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        keywords = df.iloc[:, 0].dropna().unique().tolist()
        st.write(f"Loaded {len(keywords)} keywords from your file")
else:
    keywords_text = st.text_area("Paste keywords (one per line)", height=200)
    if keywords_text:
        keywords = [k.strip() for k in keywords_text.split('\n') if k.strip()]
        st.write(f"Found {len(keywords)} keywords")

# Process keywords
if 'keywords' in locals() and len(keywords) > 0:
    # Clustering options
    st.subheader("Clustering Options")
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=min(20, len(keywords)//2), value=5)
    
    with col2:
        cluster_method = st.selectbox("Select clustering method", ["KMeans", "DBSCAN"])
    
    # Process button
    if st.button("Cluster Keywords"):
        with st.spinner("Clustering keywords..."):
            # Generate embeddings
            embeddings = model.encode(keywords, convert_to_tensor=True)
            
            # Perform clustering
            if cluster_method == "KMeans":
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings)
            else:
                from sklearn.cluster import DBSCAN
                eps = 1.0
                dbscan = DBSCAN(eps=eps, min_samples=2)
                labels = dbscan.fit_predict(embeddings)
            
            # Group keywords
            clustered = defaultdict(list)
            for i, label in enumerate(labels):
                clustered[f"Cluster {label+1}"].append(keywords[i])
            
            # Display results
            st.subheader("Clustered Keywords")
            
            # Create a DataFrame for display
            result_df = pd.DataFrame(columns=["Cluster", "Keywords"])
            for cluster, cluster_keywords in clustered.items():
                result_df = pd.concat([result_df, pd.DataFrame({
                    "Cluster": [cluster] * len(cluster_keywords),
                    "Keywords": cluster_keywords
                })], ignore_index=True)
            
            # Display as a table
            st.dataframe(result_df)
            
            # Visualization
            st.subheader("Cluster Visualization")
            
            # Dimensionality reduction for visualization
            tsne = TSNE(n_components=2, random_state=42)
            reduced = tsne.fit_transform(embeddings)
            
            # Create a DataFrame for plotting
            plot_df = pd.DataFrame({
                "x": reduced[:, 0],
                "y": reduced[:, 1],
                "cluster": [f"Cluster {label+1}" for label in labels],
                "keyword": keywords
            })
            
            # Plot with Plotly
            fig = px.scatter(plot_df, x="x", y="y", color="cluster", hover_data=["keyword"])
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="keyword_clusters.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = str(dict(clustered))
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="keyword_clusters.json",
                    mime="application/json"
                )