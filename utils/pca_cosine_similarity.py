from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def compute_pca(vectors, n_components=2):
    if len(vectors) < 2:
        raise ValueError("You must input at least 2 sentences for PCA visualization.")
    pca = PCA(n_components=n_components)
    return pca.fit_transform(vectors)

def compute_cosine_similarity(vectors):
    return cosine_similarity(vectors)
