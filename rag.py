import numpy as np
from embeddings import get_embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, documents, top_k=3):
    
    query_embedding = get_embedding(query)

    scores = []

    for doc in documents:
        score = cosine_similarity(query_embedding, doc["embedding"])
        scores.append((score, doc))

    # Sort by highest similarity
    scores.sort(key=lambda x: x[0], reverse=True)

    # Return top documents
    top_docs = [doc for score, doc in scores[:top_k]]

    return top_docs