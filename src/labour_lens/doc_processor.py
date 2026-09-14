from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

import os
import hashlib


def calculate_file_hash(file_path):
    with open(file_path, "rb") as file:
        file_contents = file.read()

    return hashlib.sha256(file_contents).hexdigest()


def process_documents():

    # --------------------------------------------------
    # 1. Load environment variables
    # --------------------------------------------------
    load_dotenv()

    # --------------------------------------------------
    # 2. Connect to Pinecone
    # --------------------------------------------------
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "labour-lens"

    pc_index = pc.Index(index_name)  # selecting existing pinecone index

    # --------------------------------------------------
    # 3. Create embeddings model
    # --------------------------------------------------
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # --------------------------------------------------
    # 4. Text splitter
    # --------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50
    )

    # --------------------------------------------------
    # 5. Find all PDF files in data folder
    # --------------------------------------------------

    # works backwards from the location of doc_processor.py to find the project root.
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    data_folder = os.path.join(
        project_root,
        "data"
    )

    # Looks through the data folder and gives me every PDF.
    pdf_files = [
        file for file in os.listdir(data_folder)
        if file.lower().endswith(".pdf")
    ]
    print(f"Found {len(pdf_files)} PDF file(s).")

    # --------------------------------------------------
    # 6. Process each PDF
    # --------------------------------------------------

    for pdf_file in pdf_files:
        # creates the full path to the file
        pdf_path = os.path.join(
            data_folder,
            pdf_file
        )

        print("\n----------------------------------------")
        print(f"Processing: {pdf_file}")
        print("----------------------------------------")

        # --------------------------------------------------
        # 7. Create document ID
        # --------------------------------------------------

        document_id = calculate_file_hash(pdf_path)

        # --------------------------------------------------
        # 8. Check whether document already exists
        # --------------------------------------------------
        first_chunk_id = f"{document_id}_chunk_0"
        # ask pinecone 'does a vector with this ID already exist?'
        existing_vectors = pc_index.fetch(
            ids=[first_chunk_id]
        )

        # if pinecone return that vector, skip the ingestion.
        if existing_vectors.vectors:
            print("Document already exists in Pinecone.")
            print("Skipping ingestion.")

            continue

        # --------------------------------------------------
        # 9. Load PDF
        # --------------------------------------------------
        print("Document is new.")
        print("Loading PDF...")

        loader = PyPDFLoader(pdf_path)

        pages = loader.load()

        # --------------------------------------------------
        # 10. Split document into chunks
        # --------------------------------------------------
        chunks = text_splitter.split_documents(pages)

        print(f"Created {len(chunks)} chunks.")

        # --------------------------------------------------
        # 11. Add metadata
        # --------------------------------------------------

        for chunk in chunks:
            chunk.metadata["document_id"] = document_id
            chunk.metadata["source"] = pdf_file

        # --------------------------------------------------
        # 12. Create deterministic chunk IDs
        # --------------------------------------------------

        # These are called deterministic IDs because you can
        # reproduce them from the same document and chunk position.
        chunk_ids = [
            f"{document_id}_chunk_{i}"
            for i in range(len(chunks))
        ]

        # --------------------------------------------------
        # 13. Upload to Pinecone
        # --------------------------------------------------

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            index_name=index_name,
            ids=chunk_ids
        )

        print("Document successfully embedded and uploaded.")
        print(f"Uploaded {len(chunks)} vectors.")

    print("\n========================================")
    print("Document processing complete.")
    print("========================================")


if __name__ == "__main__":
    process_documents()
