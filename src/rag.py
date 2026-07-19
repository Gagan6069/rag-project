from src.services.rag_service import RAGService
from src.utils.startup_validator import StartupValidator

# def print_sources(results):

#     print("\n" + "=" * 60)
#     print("SOURCES")
#     print("=" * 60)

#     if not results:
#         print("No relevant sources found.")
#         return

#     for index, result in enumerate(results, start=1):

#         print(f"\nSource #{index}")
#         print(f"File  : {result.source}")
#         print(f"Page  : {result.page}")
#         print(f"Score : {result.score:.4f}")
#         print("-" * 60)


def main():

    StartupValidator.validate_for_chat()

    rag = RAGService()

    print("=" * 60)
    print("        AI Knowledge Assistant")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Ask: ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question:
            continue

        response = rag.ask(question)

        print("\n" + "=" * 60)
        print("QUESTION")
        print("=" * 60)
        print(response.question)

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(response.answer)

        # print_sources(response.sources)


if __name__ == "__main__":
    main()