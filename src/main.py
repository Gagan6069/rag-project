from services.rag_service import RAGService


def print_sources(sources):

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for doc, score in sources:

        print(f"Page      : {doc.metadata.get('page')}")
        print(f"Source    : {doc.metadata.get('source')}")
        print(f"Score     : {score:.4f}")

        print("-" * 60)


def main():

    rag = RAGService()

    print("\nRAG Chat Started")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("Ask: ")

        if question.lower() == "exit":
            break

        response = rag.ask(question)

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(response["answer"])

        print_sources(response["sources"])


if __name__ == "__main__":
    main()