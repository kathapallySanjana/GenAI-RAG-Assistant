import json
import random

def get_embedding(text):
    # create fake embedding vector
    return [random.random() for _ in range(100)]

def load_documents():
    with open("docs.json") as f:
        docs = json.load(f)

    embeddings = []

    for doc in docs:
        emb = get_embedding(doc["content"])

        embeddings.append({
            "title": doc["title"],
            "content": doc["content"],
            "embedding": emb
        })

    return embeddings