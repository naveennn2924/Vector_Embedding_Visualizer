#  Sentence Vector Space Explorer using BAAI Embeddings

This project explores how sentence embeddings behave in vector space using the `BAAI/bge-small-en` model. It allows users to dynamically input sentences and observe how they cluster based on **cosine similarity** and **Euclidean distance**.

It uses dimensionality reduction (PCA) to visualize these high-dimensional embeddings and highlight how semantically similar sentences group together in 2D space.

---

## 🔬 Objective

- Investigate **semantic similarity** encoded in sentence vectors
- Visualize how **BAAI embedding models** position related sentences
- Show differences between **cosine similarity** and **Euclidean distance** in clustering
- Provide a tool to interactively explore embedding spaces using dynamic text input

---

## 🧠 About the Model: Is BAAI Hybrid?

This project uses the [`BAAI/bge-small-en`](https://huggingface.co/BAAI/bge-small-en) model — a **semantic embedding model** trained using contrastive learning.

- It **does not use hybrid sparse+dense search**
- It is focused on **dense vector representations** for semantically similar text
- You can use it in both **retrieval (RAG)** and **similarity-based clustering**

For hybrid retrieval (dense + sparse), models like ColBERT or hybrid rerankers are used — not BAAI bge-small-en.

---

## 💡 Example Insights

### 1. Cosine Similar Sentences Cluster Together

→ The first two cluster together in PCA space, while the third is far apart.

### 2. Euclidean Distance vs Cosine

Even when vectors are close in Euclidean space, they might be far in **cosine distance**, which better reflects semantic similarity for normalized embeddings.

---

## 📈 Visualization Outputs

- Interactive 2D scatter plot via Plotly (PCA)
- Cosine similarity matrix (range: -1 to 1)
- Optional: Euclidean distance matrix

---

## 📊 Distance Metrics Explained

| Metric | Suitable for        | Range     | Normalization Needed? |
|--------|---------------------|-----------|------------------------|
| Cosine Similarity | Semantic similarity | -1 to 1 | ✅ Yes (unit norm)     |
| Euclidean Distance | Geometric closeness | 0 to ∞ | ❌ No (sensitive to scale) |

---

## 🔗 Technologies Used

- `sentence-transformers` (BAAI embedding)
- `scikit-learn` (PCA, cosine, Euclidean)
- `plotly` (interactive 2D plotting)
- `Streamlit` (dynamic user interface)

---

## 🚀 Try It Locally

```bash
git clone https://github.com/naveennn2924/Vector_Embedding_Visualizer

pip install -r requirements.txt

python scripts/download_model.py
streamlit run app.py

