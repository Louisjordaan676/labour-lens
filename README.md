# Labour Lens

Labour Lens is a South African labour-law AI assistant designed to help users find and understand information contained in authoritative South African labour-law documents.

The project is being developed as an AI engineering portfolio project, with an emphasis on retrieval-augmented generation (RAG), source attribution, software engineering practices, and responsible use of AI for legal-information retrieval.

## Project Status

**Current phase:** Development environment setup

The application itself has not yet been implemented.

The current focus is establishing a clean, reproducible Python development environment and professional project structure.

## Planned Architecture

The eventual system is planned to follow this general architecture:

```text
Authoritative Labour-Law PDFs
            ↓
     Document Loading
            ↓
        Chunking
            ↓
       Embeddings
            ↓
     Local Vector Store
            ↓
        Retrieval
            ↓
       OpenAI LLM
            ↓
 Answer + Legal Sources
```

The initial implementation will use a local vector store while keeping the application architecture flexible enough to support migration to a managed/cloud vector database in the future.

## Project Structure

```text
labour-lens/
│
├── src/
│   └── labour_lens/
│       └── __init__.py
│
├── data/
│   └── .gitkeep
│
├── tests/
│   └── __init__.py
│
├── .gitignore
├── .env.example
├── README.md
└── requirements.txt
```

## Technology Stack

The project is currently being developed with:

* Python
* LangChain
* OpenAI API
* Git
* GitHub
* VS Code

Additional technologies will be introduced as the application is developed.

## Development Environment

The project uses a Python virtual environment to isolate its dependencies.

Python version currently used:

```text
Python 3.13.9
```

## Important Note

Labour Lens is intended to provide information retrieved from authoritative legal sources. It is not intended to replace professional legal advice.

## License

License to be determined.
