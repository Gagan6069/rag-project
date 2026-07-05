class ContextBuilder:

    @staticmethod
    def build(results):

        context = ""

        for i, (doc, score) in enumerate(results, start=1):

            context += f"""
Document {i}

Source:
{doc.metadata.get("source")}

Page:
{doc.metadata.get("page")}

Similarity Score:
{score}

Content:
{doc.page_content}

----------------------------------------------------

"""

        return context