import os
from groq import Groq
from config import GROQ_MODEL
from llms.base import BaseLLM


class GroqLLM(BaseLLM):

    def __init__(self):

        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

        # self.model = "llama3-8b-8192"
        self.model = GROQ_MODEL

    def ask(self, prompt):

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            return response.choices[0].message.content

        except AuthenticationError:
            raise RuntimeError(
                "Invalid GROQ_API_KEY."
            )

        except BadRequestError as e:
            raise RuntimeError(
                f"Groq rejected the request: {e}"
            )