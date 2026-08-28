from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# load the document
loader = PyPDFLoader(
    r"C:\Users\jorda\Projects\labour-lens\data\Basic Conditions of Employment Act [No. 75 of 1997].pdf")
pages = loader.load()

# create a text recursive splitter (chunking)

r_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=50)
chunks = r_text_splitter.split_documents(pages)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# embed_exp = embeddings_model.embed_query(chunks[0].page_content)

# new_chunks = [item.page_content for item in chunks]

# result = embeddings_model.embed_documents(new_chunks)
# # print(len(result)) # 197 because thats how many chunks are in "chunks"
# # print(len(result[0])) # 1536 , every string has this amount of vectors

vector_store = Chroma(
    collection_name="basic_conditions_of_employment",
    embedding_function=embeddings_model,
    persist_directory="./chroma_langchain_dh"
)
