from sentence_transformers import SentenceTransformer

model_name = 'BAAI/bge-small-en'
model = SentenceTransformer(model_name)
model.save('./models/bge-small-en')
print(f"Model saved to ./models/bge-small-en")
