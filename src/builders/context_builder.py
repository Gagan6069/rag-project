class ContextBuilder:

    def build(self, results):

        context = ""

        for index, result in enumerate(results, start=1):

            context += f"""
Document {index}

Source:
{result.source}

Page:
{result.page}

Similarity Score:
{result.similarity_score:.4f}

Content:
{result.page_content}

------------------------------------------------------------

"""

        return context