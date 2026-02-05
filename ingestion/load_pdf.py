from pypdf import PdfReader

PDF_PATH = "D:\\Downloads\\rag-project\\data\\aws-overview.pdf"

def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append({
                "page": page_number,
                "text": text.strip()
            })

    return documents

if __name__ == "__main__":
    docs = load_pdf(PDF_PATH)

    print(f"Total pages extracted: {len(docs)}")
    print("_"* 60)
    print("Sample page output:\n")

    print(f"Page Number: {docs[0]['page']}")
    print(docs[0]['text'][:1000]) #first 1000 chars