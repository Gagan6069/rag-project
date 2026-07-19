import csv
import json
import os
import time
from pathlib import Path

from src.services.rag_service import RAGService


class RAGEvaluator:

    def __init__(self):

        self.rag = RAGService()

    def load_benchmark(self, file_path):

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run(
        self,
        benchmark_file="src/evaluation/benchmark_questions.json",
        output_file="../outputs/run_results_v2.csv"
    ):

        benchmark = self.load_benchmark(benchmark_file)

        rows = []

        for item in benchmark:

            question = item["question"]

            expected_pages = item.get("expected_pages", [])

            print(f"\nRunning: {question}")

            start_time = time.time()

            response = self.rag.ask(question)

            end_time = time.time()

            latency = round(end_time - start_time, 3)

            sources = response.sources

            retrieved_pages = [
                source.page
                for source in sources
            ]

            retrieved_chunk_ids = [
                source.chunk_id
                for source in sources
            ]

            retrieval_methods = [
                source.retrieval_method
                for source in sources
            ]

            faiss_scores = [
                round(source.score, 4)
                for source in sources
            ]

            bm25_scores = [
                round(source.bm25_score, 4)
                for source in sources
            ]

            rrf_scores = [
                round(source.rrf_score, 4)
                for source in sources
            ]

            rerank_scores = [
                round(source.rerank_score, 4)
                for source in sources
            ]

            top_page = retrieved_pages[0] if retrieved_pages else None

            page_hit = self._page_hit(
                retrieved_pages,
                expected_pages
            )

            recall_at_k = self._recall_at_k(
                retrieved_pages,
                expected_pages
            )

            top_chunk_preview = ""

            if sources:
                top_chunk_preview = sources[0].page_content.replace(
                    "\n",
                    " "
                )[:500]

            rows.append({
                "Question": question,
                "Answer": response.answer,
                "Expected Pages": expected_pages,
                "Retrieved Pages": retrieved_pages,
                "Top Page": top_page,
                "Page Hit": page_hit,
                "Recall@K": recall_at_k,
                "Retrieved Chunk IDs": retrieved_chunk_ids,
                "Retrieval Methods": retrieval_methods,
                "FAISS Scores": faiss_scores,
                "BM25 Scores": bm25_scores,
                "RRF Scores": rrf_scores,
                "Rerank Scores": rerank_scores,
                "Top Chunk Preview": top_chunk_preview,
                "Context Length": len(response.context),
                "Prompt Length": len(response.prompt),
                "Answer Length": len(response.answer),
                "Latency (sec)": latency,
            })

        self.save_csv(rows, output_file)

    def _page_hit(self, retrieved_pages, expected_pages):

        if not expected_pages:
            return "N/A"

        return any(
            page in expected_pages
            for page in retrieved_pages
        )

    def _recall_at_k(self, retrieved_pages, expected_pages):

        if not expected_pages:
            return "N/A"

        expected_set = set(expected_pages)
        retrieved_set = set(retrieved_pages)

        hits = expected_set.intersection(retrieved_set)

        return round(
            len(hits) / len(expected_set),
            4
        )

    def save_csv(self, rows, output_file):

        output_path = Path(output_file)

        os.makedirs(
            output_path.parent,
            exist_ok=True
        )

        with open(output_path, "w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Question",
                    "Answer",
                    "Expected Pages",
                    "Retrieved Pages",
                    "Top Page",
                    "Page Hit",
                    "Recall@K",
                    "Retrieved Chunk IDs",
                    "Retrieval Methods",
                    "FAISS Scores",
                    "BM25 Scores",
                    "RRF Scores",
                    "Rerank Scores",
                    "Top Chunk Preview",
                    "Context Length",
                    "Prompt Length",
                    "Answer Length",
                    "Latency (sec)",
                ]
            )

            writer.writeheader()

            writer.writerows(rows)

        print(f"\nResults saved to {output_path}")