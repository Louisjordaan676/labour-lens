from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

pdf_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "Basic Conditions of Employment Act [No. 75 of 1997].pdf"
    )
)

loader = PyPDFLoader(pdf_path)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(pages)

print(f"Total pages: {len(pages)}")
print(f"Total chunks: {len(chunks)}")

print("\n" + "=" * 80)
print("CHUNKS CONTAINING 'OVERTIME'")
print("=" * 80)

for i, chunk in enumerate(chunks):

    if "overtime" in chunk.page_content.lower():

        print(f"\n{'-' * 80}")
        print(f"CHUNK {i}")
        print(f"PAGE: {chunk.metadata.get('page_label')}")
        print(f"CHARACTERS: {len(chunk.page_content)}")
        print("-" * 80)

        print(chunk.page_content)
