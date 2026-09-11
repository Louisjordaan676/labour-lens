from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
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

# =========== Prompt template ==========
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

# =========== document chain ==========
document_chain = create_stuff_documents_chain(
    llm,
    prompt
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
# test_docs = retriever.invoke("how many hours overtime am i allowed to work?")

# for doc in test_docs:
#     print("\n--- RETRIEVED DOCUMENT ---")
#     print("Content:", doc.page_content)
#     print("Metadata:", doc.metadata)
# =========== RAG chain ==========

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# =========== Response ==========

while True:
    user_input = input("You: ")

    if user_input == "exit":
        break
    else:
        response = rag_chain.invoke({
            "input": user_input
        })
        print(f"AI: {response["answer"]}")
        print("\nRetrieved sources:")
        source_count = 1
        for doc in response["context"]:

            print(f"Source {source_count}")
            print(f"File name: {doc.metadata["source"]}")
            print(f"Page: {doc.metadata["page_label"]}")
            source_count += 1

# # question = "How many hours may an employee work per week?"

# # documents = retriever.invoke(question)

# # for document in documents:
# #     print("\n------------------------------")
# #     print(document.page_content[:500])
#     print(document.metadata)
