from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

import os

load_dotenv()

# ==========================
# Embeddings
# ==========================

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# ==========================
# Vector store
# ==========================

vector_store = PineconeVectorStore(
    index_name="labour-lens",
    embedding=embeddings_model
)

# ==========================
# Retriever
# ==========================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# ==========================
# Test question
# ==========================

question = "how many hours overtime am i allowed to work?"

# ==========================
# Retrieve documents
# ==========================

documents = retriever.invoke(question)

# ==========================
# Display results
# ==========================

print("=" * 80)
print("RETRIEVAL RESULTS")
print("=" * 80)

print(f"\nQuestion: {question}")
print(f"Documents retrieved: {len(documents)}")

for i, doc in enumerate(documents, start=1):

    print("\n" + "-" * 80)
    print(f"RESULT {i}")
    print("-" * 80)

    print(f"File: {doc.metadata.get('source')}")
    print(f"Page: {doc.metadata.get('page_label')}")
    print(f"Document ID: {doc.metadata.get('document_id')}")
    print(f"Characters: {len(doc.page_content)}")

    print("\nCONTENT:")
    print(doc.page_content)
