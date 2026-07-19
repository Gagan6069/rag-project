from src.services.rag_service import RAGService
from debug.rag_dashboard import RAGDashboard


rag = RAGService()

while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    RAGDashboard.question(question)

    rag.ask(question)