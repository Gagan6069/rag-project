from config import DEBUG


class RAGDashboard:

    @staticmethod
    def line():
        print("=" * 90)

    @staticmethod
    def section(title):

        if not DEBUG:
            return

        print()
        RAGDashboard.line()
        print(title.upper())
        RAGDashboard.line()

    @staticmethod
    def question(question):

        if not DEBUG:
            return

        RAGDashboard.section("Question")
        print(question)

    @staticmethod
    def retrieval(results):

        if not DEBUG:
            return

        RAGDashboard.section("Retrieved Chunks")

        if not results:
            print("No chunks retrieved.")
            return

        for index, result in enumerate(results, start=1):

            print(f"Rank #{index}")
            print(f"Method       : {result.retrieval_method}")
            print(f"Chunk ID     : {result.chunk_id}")
            print(f"Page         : {result.page}")
            print(f"Source       : {result.source}")
            print(f"FAISS Score  : {result.score:.4f}")
            print(f"BM25 Score   : {result.bm25_score:.4f}")
            print(f"RRF Score    : {result.rrf_score:.4f}")
            print(f"Rerank Score : {result.rerank_score:.4f}")

            preview = result.page_content.replace("\n", " ")

            max_preview = 500

            if len(preview) > max_preview:
                preview = preview[:max_preview] + "..."

            print("\nPreview:")
            print(preview)

            print("-" * 90)

    @staticmethod
    def context(context):

        if not DEBUG:
            return

        RAGDashboard.section("Context Sent To LLM")
        print(context)

    @staticmethod
    def prompt(prompt):

        if not DEBUG:
            return

        RAGDashboard.section("Prompt")
        print(prompt)

    @staticmethod
    def answer(answer):

        if not DEBUG:
            return

        RAGDashboard.section("LLM Answer")
        print(answer)