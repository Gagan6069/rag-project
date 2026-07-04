from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector = embedding.embed_query(
    "What is leave policy?"
)

print(len(vector))

print(vector[:10])