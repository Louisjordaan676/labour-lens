from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = "labour-lens"

# =========== embeddings model ==========
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# =========== LLM ==========

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# =========== Create vector store ==========
vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings_model
)

# =========== turn vector store into retriever ==========

retriever = vector_store.as_retriever(
    # k=5 means for every question, the retriever will return top 5 chunks. can play around with this later.
    search_kwargs={"k": 5}
)

# question = "How many hours may an employee work per week?"

# documents = retriever.invoke(question)

# for document in documents:
#     print("\n------------------------------")
#     print(document.page_content[:500])
#     print(document.metadata)
