import time
import csv
import os

from services.rag_service import RAGService


class RAGEvaluator:

    def __init__(self):

        self.rag = RAGService()

    def run(self, question_list, output_file="outputs/run_results.csv"):

        results = []

        for q in question_list:

            print(f"\nRunning: {q}")

            start_time = time.time()

            response = self.rag.ask(q)

            end_time = time.time()

            latency = round(end_time - start_time, 3)

            # Collect retrieved sources summary
            sources = response.sources

            retrieved_pages = [
                str(s.page) for s in sources
            ]

            retrieved_scores = [
                round(s.score, 4) for s in sources
            ]

            results.append([
                q,
                response.answer,
                retrieved_pages,
                retrieved_scores,
                latency
            ])

        self.save_csv(results, output_file)

    def save_csv(self, data, output_file):

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow([
                "Question",
                "Answer",
                "Retrieved Pages",
                "Scores",
                "Latency (sec)"
            ])

            writer.writerows(data)

        print(f"\nResults saved to {output_file}")