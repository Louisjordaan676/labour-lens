from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.
import os

# create embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# create LLm
llm = ChatOpenAI(model="gpt-4o-mini")

system_prompt = """
You are a helpful labour law assistant. 
you will recieve context and a question to help you answer the qustion.
Do not make anything up. if you do not know the answer,
reply with 'I do not know the answer to your question'.

Context: {context}
"""

# create prompt template
prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("human", "{question}")
])


if os.path.exists("./chroma_langchain_dh"):
    vector_store = Chroma(embedding_function=embeddings_model,
                          collection_name="basic_conditions_of_employment",
                          persist_directory="./chroma_langchain_dh")
else:
    # load the document
    loader = PyPDFLoader(
        r"C:\Users\jorda\Projects\labour-lens\data\Basic Conditions of Employment Act [No. 75 of 1997].pdf")
    pages = loader.load()

    # create a text recursive splitter (chunking)
    r_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=50)
    chunks = r_text_splitter.split_documents(pages)

    # Create vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        collection_name="basic_conditions_of_employment",
        persist_directory="./chroma_langchain_dh"
    )

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

response = rag_chain.invoke({"input": input("Please enter a question: ")})
print(response["answer"])
