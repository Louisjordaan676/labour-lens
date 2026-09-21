from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
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
# 2. LLM
# ==============================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ==============================
# 3. Prompt
# ==============================

prompt = ChatPromptTemplate.from_template("""
You are LabourLens, a South African labour-law assistant.

Answer the user's question using only the provided context below.
Do not draw on outside knowledge. If the context does not contain
enough information to answer the question, respond with exactly:
"I do not know the answer to this question."

Where your answer has significant legal consequences, remind the
user to consult a qualified South African labour attorney.

Context: {context}
Question: {input}
""")

# ==============================
# 4. Document chain
# ==============================

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# ==============================
# 5. Vector store
# ==============================

vector_store = PineconeVectorStore(
    index_name="labour-lens",
    embedding=embeddings_model
)

# ==============================
# 6. Retriever
# ==============================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# ==============================
# 7. RAG chain
# ==============================

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# ==============================
# 8. Question
# ==============================

question = "What is the South African minimum wage?"
# ==============================
# 9. Invoke RAG chain
# ==============================

response = rag_chain.invoke({
    "input": question
})

# ==============================
# 10. Display result
# ==============================

print("=" * 80)
print("FULL RAG CHAIN TEST")
print("=" * 80)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(response["answer"])

print("\nRETRIEVED DOCUMENTS:")
print(f"{len(response['context'])} documents retrieved")

for i, doc in enumerate(response["context"], start=1):
    print("\n" + "-" * 80)
    print(f"DOCUMENT {i}")
    print("-" * 80)
    print(f"File: {doc.metadata.get('source')}")
    print(f"Page: {doc.metadata.get('page_label')}")
