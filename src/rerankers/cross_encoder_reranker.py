from sentence_transformers import CrossEncoder

from rerankers.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )


    def rerank(self, question, results):

        pairs = []

        for result in results:

            pairs = [
                (question, result.page_content)
                for result in results
            ]


        scores = self.model.predict(pairs)


        reranked = []


        for result, score in zip(results, scores):

            reranked.sort(
                key=lambda x: x.rerank_score,
                reverse=True
            )

            reranked.append(result)


        reranked.sort(
            key=lambda x: x.rerank_score,
            reverse=True
        )


        return reranked