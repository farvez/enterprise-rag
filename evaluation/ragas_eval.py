import os
from dotenv import load_dotenv
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)

# 1. Setup the environment path 
env_path = Path(__file__).parent.parent / "query" / ".env"
load_dotenv(dotenv_path=env_path)

# 2. Initialize Groq via LangChain 
groq_chat = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
evaluator_llm = LangchainLLMWrapper(groq_chat)

# 3. Initializing Embeddings via LangChain
lc_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
evaluator_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)


data = {
    "question": ["What is EC2?", "What is AWS Bedrock?"],
    "answer": [
        "Amazon EC2 provides resizable compute capacity in the cloud.",
        "AWS Bedrock is a managed service for foundation models."
    ],
    "contexts": [
        ["Amazon EC2 provides secure, resizable compute capacity..."],
        ["AWS Bedrock makes foundation models available through an API..."]
    ],
    "reference": [
        "Amazon Elastic Compute Cloud (EC2) is a web service that provides secure, resizable compute capacity.",
        "AWS Bedrock is a fully managed service that offers high-performing foundation models."
    ]
}
dataset = Dataset.from_dict(data)

# 4. Attach models to metrics
faithfulness.llm = evaluator_llm
answer_relevancy.llm = evaluator_llm
answer_relevancy.embeddings = evaluator_embeddings
context_precision.llm = evaluator_llm

# 5. Running the evaluation

from ragas.run_config import RunConfig
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    run_config=RunConfig(max_workers=1) 
)

print("\n--- Final Evaluation Results ---")
print(results)