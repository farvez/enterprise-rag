from sentence_transformers import CrossEncoder

#lightweight,fast,easy to use
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANK_MODEL_NAME)

    def rerank(self, question:str, chunks: list, top_n: int =3):
        """
        Re-rank retrieved chunks based on relevance to the question
        """

        pairs = [
            (question, chunk["text"])
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        #Attach score
        for chunk , score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        #sort by score
        reranked = sorted(
            chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_n]