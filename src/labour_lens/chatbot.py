from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeSparseVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = "labour-lens"

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# =========== Create vector store ==========
vector_store = PineconeSparseVectorStore(
    index_name=index_name,
    embedding=embeddings_model
)

# =========== turn vector store into retriever ==========

retriever = vector_store.as_retriever(
    # k=5 means for every question, the retriever will return top 5 chunks. can play around with this later.
    search_kwargs={"k": 5}
)
