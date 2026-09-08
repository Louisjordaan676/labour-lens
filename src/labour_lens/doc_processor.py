from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import hashlib


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# 2. Connect to Pinecone
# --------------------------------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "labour-lens"

pc_index = pc.Index(index_name)

# --------------------------------------------------
# 3. Document we want to ingest
# --------------------------------------------------
pdf_path = r"C:\Users\jorda\Projects\labour-lens\data\Basic Conditions of Employment Act [No. 75 of 1997].pdf"

file_name = os.path.basename(pdf_path)

# --------------------------------------------------
# 4. Create a stable document ID
# --------------------------------------------------

document_id = hashlib.sha256(
    file_name.encode("utf-8")
).hexdigest()

# --------------------------------------------------
# 5. Check whether this document already exists
# --------------------------------------------------

existing_vectors = pc_index.query(
    vector=[0.0] * 1536,
    top_k=1,
    include_metadata=True,
    filter={
        "document_id": {"$eq": document_id}
    }
)

if existing_vectors["matches"]:
    print("Document already exists in Pinecone.")
    print("Skipping ingestion.")

else:
    print("Document not found in Pinecone.")
    print("Starting ingestion...")

    # --------------------------------------------------
    # 6. Load PDF
    # --------------------------------------------------

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # --------------------------------------------------
    # 7. Split document into chunks
    # --------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(pages)

    print(f"Created {len(chunks)} chunks.")

    # --------------------------------------------------
    # 8. Add metadata
    # --------------------------------------------------

    for chunk in chunks:

        chunk.metadata["document_id"] = document_id
        chunk.metadata["source"] = file_name

    # --------------------------------------------------
    # 9. Create deterministic IDs for chunks
    # --------------------------------------------------
    chunk_ids = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{document_id}_chunk_{i}"
        chunk_ids.append(chunk_id)

    # --------------------------------------------------
    # 10. Create embeddings model
    # --------------------------------------------------
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # --------------------------------------------------
    # 11. Upload to Pinecone
    # --------------------------------------------------
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        index_name=index_name,
        ids=chunk_ids
    )
    print("Document successfully embedded and uploaded to Pinecone.")

    print(f"Uploaded {len(chunks)} vectors.")


# # create LLm
# llm = ChatOpenAI(model="gpt-4o-mini")

# system_prompt = """
# You are a helpful labour law assistant.
# you will recieve context and a question to help you answer the qustion.
# Do not make anything up. if you do not know the answer,
# reply with 'I do not know the answer to your question'.

# Context: {context}
# """

# # create prompt template
# prompt = ChatPromptTemplate([
#     ("system", system_prompt),
#     ("human", "{input}")
# ])


# # load the document
# loader = PyPDFLoader(
#     r"C:\Users\jorda\Projects\labour-lens\data\Basic Conditions of Employment Act [No. 75 of 1997].pdf")
# pages = loader.load()

# # create a text recursive splitter (chunking)
# r_text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=700, chunk_overlap=50)


# chunks = r_text_splitter.split_documents(pages)

# vector_store = PineconeVectorStore.from_documents(
#     documents=chunks,
#     embedding=embeddings_model,
#     index_name="labour-lens"
# )


# retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# combine_docs_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# response = rag_chain.invoke({"input": input("Please enter a question: ")})
# print(response["answer"])
