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

FAISS Score:
{result.score:.4f}

BM25 Score:
{result.bm25_score:.4f}

RRF Score:
{result.rrf_score:.4f}

Rerank Score:
{result.rerank_score:.4f}

Retrieval Method:
{result.retrieval_method}

Content:
{result.page_content}

------------------------------------------------------------

"""

        return context