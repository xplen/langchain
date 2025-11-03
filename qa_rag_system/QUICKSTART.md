# Quick Start Guide

Get up and running with the Q&A RAG System in minutes!

## Step 1: Install Dependencies

```bash
cd qa_rag_system
pip install -e .
```

## Step 2: Configure Environment

Copy the example environment file:
```bash
cp env.example .env
```

Edit `.env` with your settings. For a quick start with **local/open-source** options:

```env
# Use Ollama for LLM (install Ollama first: https://ollama.ai)
LLM_PROVIDER=ollama
LLM_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434

# Use Sentence Transformers for embeddings (free, local)
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Use Chroma for vector DB (free, local)
VECTOR_DB=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

Or use **OpenAI** (requires API keys):

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=your_key_here

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

VECTOR_DB=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

## Step 3: Prepare Documents

Create a `documents/` folder and add your files:
```bash
mkdir documents
# Add PDFs, text files, markdown files, etc.
```

## Step 4: Run the Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Step 5: Index Documents

1. In the sidebar, choose an indexing method:
   - **Upload Files**: Upload PDFs, text files, etc.
   - **Enter Paths**: Provide file paths, directory paths, or URLs
   - **Load Existing Index**: Use a previously created index

2. Click the index button

3. Wait for indexing to complete

## Step 6: Ask Questions

Type your question in the main interface and click "Ask"!

## Example Questions

- "What is the main API endpoint for authentication?"
- "How do I handle errors in this system?"
- "Explain the configuration options"
- "What are the best practices for deployment?"

## Troubleshooting

### Ollama Not Working

1. Install Ollama: https://ollama.ai
2. Start Ollama: `ollama serve`
3. Pull a model: `ollama pull llama3`
4. Verify: `curl http://localhost:11434/api/tags`

### Sentence Transformers Slow

First run will download the model (~90MB). Subsequent runs are faster.

### Chroma DB Errors

Delete the `chroma_db` folder and re-index:
```bash
rm -rf chroma_db
```

## Next Steps

- Read the full [README.md](README.md) for advanced configuration
- Try different chunking strategies
- Experiment with different models
- Deploy with Docker (see README)

