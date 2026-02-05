import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from groq import Groq
# from rag_query import search
from query.rag_query import search

# This finds the directory where THIS file (rag_llm_m.py) is located
# Since .env is in the same 'query' folder, we use .parent
env_path = Path(__file__).resolve().parent / '.env'

print(f"DEBUG: Looking for .env at: {env_path}")
print(f"DEBUG: Does .env exist there? {env_path.exists()}")

# Load the file from the query folder
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(f"GROQ_API_KEY not found at {env_path}. Check the file name and content.")

# file_found = load_dotenv(find_dotenv())
# print(f"DEBUG: Was .env file found? {file_found}")

# api_key = os.environ.get("GROQ_API_KEY")
# print(f"DEBUG: API Key found in env: {api_key[:10] if api_key else 'None'}...")

# if not api_key:
#     raise ValueError("GROQ_API_KEY not found! Check your .env file location.")

client = Groq(api_key=api_key)

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful AWS assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

def build_prompt(context_chunks, question):
    context_text = ""

    for i, chunk in enumerate(context_chunks, start=1):
        context_text += (
            f"[source {i} | page {chunk['page']}]\n"
            f"{chunk['text']}\n\n"
        )
        prompt = f"""
        You are a helpful AWS assisatnt.

        Use ONLY the context provided below to answer the question.
        Cite the source number and page in your answer.
        If the answer is not found, say "I don't know".

        Context:
        {context_text}

        Question:
        {question}

        Answer (with citation):
        """

        return prompt
if __name__ == "__main__":
    question = "What is AWS IAM and Why it is used"

    retrieved_chunks = search(question)
    prompt = build_prompt(retrieved_chunks, question)

    print("\n--- PROMPT SENT TO LLM ---\n")
    print(prompt[:2000])

    response = call_llm(prompt)

    print("\n---- FINAL ANSWER ---\n")
    print(response)
