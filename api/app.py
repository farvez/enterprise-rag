import time
import logging
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from query.reranker import Reranker
from typing import Dict
import hashlib

from query.rag_query import search
from query.rag_llm_m import build_prompt, call_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag-api")

app = FastAPI(title="AWS RAG Assistant")

RAG_CACHE: Dict[str, dict] = {}

def cache_key(question: str) -> str:
    normalized = question.strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    key = cache_key(request.question)

    try:
        if key in RAG_CACHE:
            logger.info(f"[{request_id}]cache hit")
            cached_response = RAG_CACHE[key]

            cached_response["cache"] = "hit"
            return cached_response
            logger.info(f"[{request_id}] Cache miss")

        reranker = Reranker()
        logger.info(f"[{request_id}] Recevied question: {request.question}")
        retrieved_chunks = search(request.question)
        reranked_chunks = reranker.rerank(
            request.question,
            retrieved_chunks,
            top_n=3
        )
        logger.info(
            f"[{request_id}] Re-ranked scores:"
            f"{[(c['page'], round(c['rerank_score'], 3)) for c in reranked_chunks]}"
        )
        pages = [chunk["page"] for chunk in retrieved_chunks]
        sources = [{
            "page": chunk["page"],
            "rerank_score": round(chunk.get("rerank_score", 0), 3)
        } 
        for chunk in reranked_chunks
        ]
        logger.info(f"[{request_id}] Retrived pages: {pages}")

        prompt = build_prompt(retrieved_chunks, request.question)
        prompt_size = len(prompt)
        logger.info(f"[{request_id}] Prompt size (chars): {prompt_size}")
        answer = call_llm(prompt)
        latency =round(time.time() - start_time, 2)
        logger.info(F"[{request_id}] Request completed in {latency}")

        # sources = [
        #     {"page": chunk["page"]}
        #     for chunk in retrieved_chunks
        # ]

        response = {
            "question": request.question,
            "answer": answer,
            "sources": [{"page": p} for p in pages],
            "reranked_sources": sources,
            "latency_seconds": latency,
            "cache": "miss"
        }
        RAG_CACHE[key] = response
        return response

    except Exception as e:
        latency = round(time.time() - start_time, 2)
        logger.error(
            f"[{request_id}] Error processing question = '{request.question}'"
            f"[{request_id}] after {latency} seconds | error={str(e)}"
        )

        return {
            "error": "internal error occured while processing the request",
            "latency_seconds": latency
        }