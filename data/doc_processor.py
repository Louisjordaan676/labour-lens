from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load the document
loader = PyPDFLoader("Basic Conditions of Employment Act [No. 75 of 1997].pdf")
pages = loader.load()

# for page in range(0, len(pages), 10):
#     print(pages[page].page_content)
#     print(pages[page].metadata)

# create a text recursive splitter (chunking)

r_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, chunk_overlap=50)
chunks = r_text_splitter.split_documents(pages)
print(f"chunks length: {len(chunks)}")
