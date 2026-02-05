import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTORSTORE_PATH = "D:\\Downloads\\rag-project\\vectorstore\\faiss_index"

os.makedirs(VECTORSTORE_PATH, exist_ok=True)

def create_faiss_index(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings

def save_vectorstore(index, chunks):
    faiss.write_index(index, f"{VECTORSTORE_PATH}/index.faiss")

    with open(f"{VECTORSTORE_PATH}/metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

def build_vectorstore(chunks):
    index,embeddings = create_faiss_index(chunks)
    save_vectorstore(index,chunks)

    return{
        "index":index,
        "embeddings_count": len(embeddings),
        "chunks_count": len(chunks)
    }

if __name__ == "__main__":
    from load_pdf import load_pdf
    from chunk_text import process_documents

    docs = load_pdf("D:\\Downloads\\rag-project\\data\\aws-overview.pdf")
    chunks = process_documents(docs)

    print(f"Total chunks to embed: {len(chunks)}")

    #index, _ = create_faiss_index(chunks)
    #save_vectorstore(index,chunks)
    
    result = build_vectorstore(chunks)
    print("Vectorstore build summary")
    print(result)
    #print("FAISS index and metadats saved successfully")