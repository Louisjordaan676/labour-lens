from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

import os

load_dotenv()

# ==============================
# 1. Embeddings
# ==============================

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# ==============================
# 2. Vector store
# ==============================

vector_store = PineconeVectorStore(
    index_name="labour-lens",
    embedding=embeddings_model
)

# ==============================
# 3. Retriever
# ==============================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# ==============================
# 4. LLM
# ==============================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ==============================
# 5. Question
# ==============================

question = "how many hours overtime am i allowed to work?"

# ==============================
# 6. Retrieve documents
# ==============================

documents = retriever.invoke(question)

# ==============================
# 7. Build context
# ==============================

context = "\n\n".join(
    document.page_content
    for document in documents
)

# ==============================
# 8. Prompt LLM directly
# ==============================

prompt = f"""
You are LabourLens, a South African labour-law assistant.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge.

If the provided context does not contain enough information
to answer the question, respond with exactly:

"I do not know the answer to this question."

Context:
{context}

Question:
{question}
"""

# ==============================
# 9. Ask the LLM
# ==============================

response = llm.invoke(prompt)

# ==============================
# 10. Display result
# ==============================

print("=" * 80)
print("DIRECT LLM TEST")
print("=" * 80)

print("\nQUESTION:")
print(question)

print("\nLLM ANSWER:")
print(response.content)

print("\nRETRIEVED DOCUMENTS:")
print(f"{len(documents)} documents retrieved")
