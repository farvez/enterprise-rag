import os 
from rag_query import search

#pseudo interface - we can replace with open ai, bedrock

def call_llm(prompt: str) -> str:
    """

    Mock LLM call (for now)
    Replace with real LLM later
    """

    return "LLM RESPONSE PLACEHOLDER"

def build_prompt(context_chunks, question):
    context_text = ""

    for i, chunk in enumerate(context_chunks, start=1):
        context_text += (
        f"[source{i} | page {chunk['page']}]\n"
        f"{chunk['text']}\n\n"
        )

    prompt = f"""
    You are a helpful AWS assistant.

    Use ONLY the context provided below to answer the question.
    Cite the source number and page in your answer.
    if the answer is not present, say "I don't know".

    Context:
    {context_text}

    Question:
    {question}

    Answern (with citation):
    """
    return prompt

if __name__ == "__main__":
    question = "What is AWS IAM and why is it used?"

    retrieved_chunks = search(question)
    prompt = build_prompt(retrieved_chunks, question)

    print("\n--- PROMPT SENT TO LLM ---\n")
    print(prompt[:2000])
    
    response = call_llm(prompt)

    print("\n--- FINAL ANSWER ---\n")
    print(response)