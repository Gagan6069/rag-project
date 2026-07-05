from config import DEBUG

class RAGDashboard:
    
    @staticmethod
    def line():

        print("=" * 90)

    @staticmethod
    def section(title):

        print()
        RAGDashboard.line()
        print(title.upper())
        RAGDashboard.line()

    @staticmethod
    def question(question):

        RAGDashboard.section("Question")

        print(question)

    @staticmethod
    def retrieval(results):

        RAGDashboard.section("Retrieved Chunks")

        if not results:
            print("No chunks retrieved.")
            return

        for index, result in enumerate(results, start=1):

            print(f"Rank #{index}")
            print(f"Score : {result.score:.4f}")
            print(f"Page  : {result.page}")
            print(f"Source: {result.source}")

            preview = result.page_content.replace("\n", " ")

            MAX_PREVIEW = 500
            
            if len(preview) > MAX_PREVIEW:
                preview = preview[:MAX_PREVIEW] + "..."

            print("\nPreview:")
            print(preview)

            print("-" * 90)

    @staticmethod
    def context(context):

        RAGDashboard.section("Context Sent To LLM")

        print(context)

    @staticmethod
    def prompt(prompt):

        RAGDashboard.section("Prompt")

        print(prompt)

    @staticmethod
    def answer(answer):

        RAGDashboard.section("LLM Answer")

        print(answer)