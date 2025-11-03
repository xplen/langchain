# Q&A RAG System

A comprehensive RAG (Retrieval-Augmented Generation) system for indexing technical documentation and answering questions with cited sources. This system demonstrates RAG architecture, chunking strategies, and retrieval optimization.

## Features

- 🔍 **Multiple Vector Databases**: Support for Pinecone (cloud) and Chroma (local/open-source)
- 🤖 **Multiple LLM Providers**: OpenAI GPT-4/3.5 or Llama via Ollama (local)
- 📊 **Multiple Embedding Models**: OpenAI embeddings or Sentence-Transformers (open-source)
- 📄 **Flexible Document Loading**: PDF, text, markdown, code files, and URLs
- ✂️ **Advanced Chunking**: Recursive character and token-based chunking strategies
- 📚 **Source Citations**: Answers include citations to source documents
- 🎨 **Modern UI**: Streamlit-based web interface
- 🐳 **Docker Support**: Easy deployment with Docker and Docker Compose

## Architecture

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│   Loader    │────▶│   Chunker    │
└─────────────┘     └──────┬───────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Embeddings  │
                    └──────┬───────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Vector Store   │
                  │ (Pinecone/Chroma)│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    Retriever    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   RAG Chain     │
                  │  (LLM + Context) │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Answer + Sources│
                  └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- (Optional) Docker and Docker Compose for containerized deployment

### Local Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd qa_rag_system
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

   Or using `uv`:
   ```bash
   uv pip install -e .
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your configuration (see Configuration section below).

## Configuration

Create a `.env` file in the project root with the following configuration:

### LLM Configuration

**Option 1: OpenAI**
```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=your_openai_api_key_here
```

**Option 2: Ollama (Local)**
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
```

### Embedding Configuration

**Option 1: OpenAI Embeddings**
```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

**Option 2: Sentence Transformers (Open-source)**
```env
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Vector Database Configuration

**Option 1: Pinecone**
```env
VECTOR_DB=pinecone
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=qa-rag-index
```

**Option 2: Chroma (Local)**
```env
VECTOR_DB=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### RAG Configuration
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
RETRIEVAL_SCORE_THRESHOLD=0.7
```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### Using the Python API

```python
from qa_rag_system.app import QARAGSystem
from qa_rag_system.config import AppConfig

# Load configuration from environment
config = AppConfig.from_env()

# Initialize the system
rag_system = QARAGSystem(config)

# Index documents
rag_system.index_documents([
    "./documents/python_docs.pdf",
    "./documents/aws_guide.md",
    "https://docs.python.org/3/library/os.html"
])

# Query the system
result = rag_system.query("How do I read a file in Python?")
print(result["answer"])
print(f"Sources: {result['sources']}")
```

## Docker Deployment

### Build and Run with Docker Compose

1. **Set up your `.env` file** (see Configuration section)

2. **Build and run**:
   ```bash
   docker-compose up --build
   ```

3. **Access the app** at `http://localhost:8501`

### Build and Run with Docker

```bash
# Build the image
docker build -t qa-rag-system .

# Run the container
docker run -p 8501:8501 --env-file .env qa-rag-system
```

## Deployment Options

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Upload your code
3. Add your API keys as secrets in Space settings
4. The app will automatically deploy

### Railway

1. Connect your GitHub repository to Railway
2. Set environment variables in Railway dashboard
3. Deploy automatically on push

## Document Types Supported

- **PDF files** (`.pdf`)
- **Text files** (`.txt`, `.md`)
- **Code files** (`.py`, `.js`, `.ts`, `.html`, `.css`)
- **URLs** (web pages via WebBaseLoader)

## Chunking Strategies

### Recursive Character Text Splitter (Default)
- Splits text recursively by paragraphs, sentences, and words
- Preserves document structure
- Best for structured documents

### Token Text Splitter
- Splits by token count
- Useful for LLM token limits
- Best for code or unstructured text

## Examples

### Example 1: Index Python Documentation

```python
from qa_rag_system.app import QARAGSystem

rag_system = QARAGSystem()
rag_system.index_documents([
    "https://docs.python.org/3/library/",
    "./local_docs/python_tutorial.pdf"
])

result = rag_system.query("What is a decorator in Python?")
print(result["answer"])
```

### Example 2: Use Local Models (Ollama + Sentence Transformers)

Set in `.env`:
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DB=chroma
```

Then run:
```python
rag_system = QARAGSystem()
rag_system.index_documents(["./docs"])
result = rag_system.query("Your question here")
```

## Project Structure

```
qa_rag_system/
├── src/
│   └── qa_rag_system/
│       ├── __init__.py
│       ├── config.py          # Configuration management
│       ├── document_loader.py # Document loading and chunking
│       ├── embeddings.py      # Embedding utilities
│       ├── llm.py            # LLM utilities
│       ├── vector_store.py   # Vector store abstraction
│       ├── rag_chain.py      # RAG chain implementation
│       └── app.py            # Main application class
├── app.py                    # Streamlit frontend
├── pyproject.toml           # Project dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy src/
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Ensure your `.env` file exists and contains `OPENAI_API_KEY=your_key`

2. **"Pinecone index not found"**
   - Create the index in Pinecone dashboard first, or use Chroma for local development

3. **"Ollama connection refused"**
   - Ensure Ollama is running: `ollama serve`
   - Check `OLLAMA_BASE_URL` in your `.env`

4. **Chroma DB errors**
   - Ensure write permissions for `CHROMA_PERSIST_DIRECTORY`
   - Try deleting the directory and re-indexing

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with [LangChain](https://github.com/langchain-ai/langchain), [Streamlit](https://streamlit.io/), and other open-source libraries.

