import re
import tiktoken
from tqdm import tqdm

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

tokenizer = tiktoken.get_encoding("cl100k_base")

def clean_text(text: str) -> str:
    #remove excessive new lines
    text = re.sub(r"\n{2,}", "\n", text)

    #normalize spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str):
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text_decoded = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text_decoded)

        start = end - CHUNK_OVERLAP

    return chunks

def process_documents(documents):
    all_chunks = []

    print(f"Processing {len(documents)} pages...")
    for doc in tqdm(documents, desc="Chunking pages", unit="pages"):
        cleaned = clean_text(doc["text"])
        chunks = chunk_text(cleaned)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "page": doc["page"],
                "chunk_id": i,
                "text": chunk
            })
    return all_chunks