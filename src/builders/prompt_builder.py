from prompts import RAG_PROMPT


class PromptBuilder:

    @staticmethod
    def build(context, question):

        return RAG_PROMPT.format(

            context=context,

            question=question
        )