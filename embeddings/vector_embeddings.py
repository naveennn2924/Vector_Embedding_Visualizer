from sentence_transformers import SentenceTransformer

model = SentenceTransformer('./models/bge-small-en')  # Load from disk

def get_embeddings(sentences: list[str]):
    return model.encode(sentences, normalize_embeddings=True)
