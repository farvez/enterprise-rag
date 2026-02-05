from load_pdf import load_pdf
from chunk_text import process_documents

docs = load_pdf("D:\\Downloads\\rag-project\\data\\aws-overview.pdf")
chunks = process_documents(docs)

print(f"Total chunks created: {len(chunks)}")
print("_" * 60)
print("Sample Chunks:\n")
print(chunks[0]["text"][:100])