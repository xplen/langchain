# Deployment Guide

This guide covers deploying the Q&A RAG System to various platforms.

## Docker Deployment

### Local Docker

1. **Build the image**:
   ```bash
   docker build -t qa-rag-system .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 --env-file .env qa-rag-system
   ```

### Docker Compose

1. **Set up your `.env` file**

2. **Start services**:
   ```bash
   docker-compose up --build
   ```

3. **Access at** `http://localhost:8501`

## Hugging Face Spaces

### Step 1: Create a Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Create a new Space
3. Choose "Streamlit" SDK

### Step 2: Upload Your Code

Upload these files to your Space:
- `app.py`
- `src/` directory
- `pyproject.toml`
- `README.md`

### Step 3: Configure Secrets

In Space Settings → Secrets, add:
- `OPENAI_API_KEY` (if using OpenAI)
- `PINECONE_API_KEY` (if using Pinecone)
- Other environment variables as needed

### Step 4: Create `requirements.txt`

For Hugging Face Spaces, create `requirements.txt`:

```txt
langchain-core>=0.3.0
langchain-openai>=0.2.0
langchain-ollama>=0.2.0
langchain-chroma>=0.2.0
langchain-text-splitters>=0.3.0
langchain-community>=0.3.0
pinecone-client>=3.0.0
chromadb>=0.5.0
sentence-transformers>=2.5.0
streamlit>=1.38.0
python-dotenv>=1.0.0
pypdf>=4.0.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
requests>=2.32.0
tiktoken>=0.8.0
```

### Step 5: Deploy

Push to your Space repository. Hugging Face will automatically build and deploy.

## Railway

### Step 1: Connect Repository

1. Go to [Railway](https://railway.app)
2. Create a new project
3. Connect your GitHub repository

### Step 2: Configure Build

Railway will auto-detect Python. Create `Procfile`:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Step 3: Set Environment Variables

In Railway dashboard → Variables, add all required `.env` variables.

### Step 4: Deploy

Railway will automatically deploy on push to your connected branch.

## Render

### Step 1: Create Web Service

1. Go to [Render](https://render.com)
2. Create a new Web Service
3. Connect your repository

### Step 2: Configure

- **Build Command**: `pip install -e .`
- **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### Step 3: Set Environment Variables

Add all required variables in the Environment section.

### Step 4: Deploy

Render will automatically deploy.

## Google Cloud Run

### Step 1: Build and Push Image

```bash
# Build
docker build -t gcr.io/YOUR_PROJECT_ID/qa-rag-system .

# Push
docker push gcr.io/YOUR_PROJECT_ID/qa-rag-system
```

### Step 2: Deploy

```bash
gcloud run deploy qa-rag-system \
  --image gcr.io/YOUR_PROJECT_ID/qa-rag-system \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="OPENAI_API_KEY=your_key,PINECONE_API_KEY=your_key"
```

## AWS EC2 / ECS

### EC2 Deployment

1. **Launch EC2 instance** (Ubuntu recommended)
2. **Install dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip docker.io -y
   ```

3. **Clone and run**:
   ```bash
   git clone <your-repo>
   cd qa_rag_system
   docker-compose up -d
   ```

4. **Configure security group** to allow port 8501

### ECS Deployment

1. **Create ECR repository**
2. **Build and push image**:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t qa-rag-system .
   docker tag qa-rag-system:latest <account>.dkr.ecr.us-east-1.amazonaws.com/qa-rag-system:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/qa-rag-system:latest
   ```

3. **Create ECS task definition** with environment variables
4. **Create ECS service** with load balancer

## Environment Variables Reference

All platforms require these environment variables (adjust based on your configuration):

```env
# LLM
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=your_key

# Embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

# Vector DB
VECTOR_DB=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Or for Pinecone:
# VECTOR_DB=pinecone
# PINECONE_API_KEY=your_key
# PINECONE_ENVIRONMENT=us-east-1-aws
# PINECONE_INDEX_NAME=qa-rag-index

# RAG Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
RETRIEVAL_SCORE_THRESHOLD=0.7
```

## Production Considerations

1. **Use managed vector DB**: Pinecone or managed Chroma for production
2. **Set up monitoring**: Add logging and error tracking
3. **Scale appropriately**: Use load balancers for multiple instances
4. **Secure API keys**: Never commit `.env` files
5. **Persistent storage**: For Chroma, use persistent volumes
6. **Rate limiting**: Add rate limiting for public deployments
7. **Caching**: Consider caching frequent queries

## Troubleshooting

### Port Issues

Some platforms require using `$PORT` environment variable:
```python
import os
port = int(os.getenv("PORT", 8501))
```

### Memory Issues

For large document sets:
- Use Pinecone instead of local Chroma
- Increase container memory limits
- Use smaller chunk sizes

### Cold Starts

For serverless deployments:
- Pre-warm the application
- Use smaller models for faster startup
- Consider keeping a warm instance

