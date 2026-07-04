from langchain_ollama import ChatOllama


class LocalLLM:

    def __init__(self):

        self.llm = ChatOllama(
            # model="gemma3:4b",
            model = "llama3.2:3b",
            temperature=0
        )

    def ask(self, prompt):

        response = self.llm.invoke(prompt)

        return response.content


if __name__ == "__main__":

    llm = LocalLLM()

    answer = llm.ask("Explain Vector Database in one paragraph.")

    print(answer)