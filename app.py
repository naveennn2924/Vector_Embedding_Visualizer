import streamlit as st
import numpy as np
from embeddings.vector_embeddings import get_embeddings
from utils.pca_cosine_similarity import compute_pca, compute_cosine_similarity
from utils.plot import plot_pca_with_plotly  

st.title("Vector Space Visualizer: Semantic Clustering with BAAI Embeddings ")

input_text = st.text_area("Enter sentences (one per line):", height=200)
sentences = [s.strip() for s in input_text.splitlines() if s.strip()]

if st.button("Run") and len(sentences) >= 2:
    vectors = get_embeddings(sentences)

    if len(sentences) > 1000:
        st.warning("Please enter fewer than 1000 sentences.")
        st.stop()

    if np.isnan(vectors).any() or np.isinf(vectors).any():
        st.error("Embedding contains NaN or Inf values.")
        st.stop()

    reduced = compute_pca(vectors)
    sim_matrix = compute_cosine_similarity(vectors)

    st.subheader("📈 PCA 2D Scatter Plot ")
    st.plotly_chart(plot_pca_with_plotly(reduced, sentences))

    st.subheader("📊 Cosine Similarity Matrix")
    st.dataframe(np.round(sim_matrix, 3))
