import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


VECTORSTORE_PATH = "D:\\Downloads\\rag-project\\vectorstore\\faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-V2"
TOP_K = 5

def load_vectorstore():
    index = faiss.read_index(f"{VECTORSTORE_PATH}/index.faiss")

    with open(f"{VECTORSTORE_PATH}/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    return index, metadata

def embed_query(query:str):
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode([query])

def search(query:str):
    index, metadata = load_vectorstore()
    query_embedding = embed_query(query)

    distance, indices = index.search(
        np.array(query_embedding).astype("float32"), TOP_K
    )

    results = []

    for idx in indices[0]:
        results.append(metadata[idx])
    
    return results

if __name__ == "__main__":
    question = "What is aws iam and why it is used?"

    reterived_chunks = search(question)

    print(f"\nUser Question: {question}\n")
    print("Retreived context:\n")
    print("-" * 60)

    for i, chunk in enumerate(reterived_chunks, start =1):
        print(f"[chunk{i}] (page {chunk['page']})")
        print(chunk["text"][:500])
        print("-" * 60)




