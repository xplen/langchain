"""Document loading and processing utilities."""

import os
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


class DocumentLoader:
    """Loads documents from various sources."""

    @staticmethod
    def load_from_directory(
        directory_path: str,
        glob_pattern: str = "**/*",
        recursive: bool = True,
    ) -> list[Document]:
        """Load documents from a directory.

        Args:
            directory_path: Path to the directory containing documents.
            glob_pattern: Glob pattern to match files.
            recursive: Whether to search recursively.

        Returns:
            List of loaded documents.
        """
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            recursive=recursive,
            loader_kwargs={"encoding": "utf-8"},
            use_multithreading=True,
        )
        return loader.load()

    @staticmethod
    def load_from_file(file_path: str) -> list[Document]:
        """Load a single document from a file.

        Args:
            file_path: Path to the file.

        Returns:
            List containing the loaded document.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        elif suffix in [".txt", ".md", ".py", ".js", ".ts", ".html", ".css"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            msg = f"Unsupported file type: {suffix}"
            raise ValueError(msg)

        return loader.load()

    @staticmethod
    def load_from_url(url: str) -> list[Document]:
        """Load documents from a URL.

        Args:
            url: URL to load documents from.

        Returns:
            List of loaded documents.
        """
        loader = WebBaseLoader(url)
        return loader.load()

    @staticmethod
    def load_from_paths(paths: list[str]) -> list[Document]:
        """Load documents from multiple paths (files, directories, or URLs).

        Args:
            paths: List of paths (files, directories, or URLs).

        Returns:
            List of all loaded documents.
        """
        all_documents: list[Document] = []

        for path in paths:
            if path.startswith(("http://", "https://")):
                docs = DocumentLoader.load_from_url(path)
            elif os.path.isfile(path):
                docs = DocumentLoader.load_from_file(path)
            elif os.path.isdir(path):
                docs = DocumentLoader.load_from_directory(path)
            else:
                msg = f"Path does not exist: {path}"
                raise ValueError(msg)

            all_documents.extend(docs)

        return all_documents


class DocumentChunker:
    """Chunks documents using various strategies."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive",
    ) -> None:
        """Initialize the document chunker.

        Args:
            chunk_size: Size of each chunk in characters or tokens.
            chunk_overlap: Overlap between chunks.
            strategy: Chunking strategy ('recursive' or 'token').
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

        if strategy == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        elif strategy == "token":
            self.splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            msg = f"Unsupported chunking strategy: {strategy}"
            raise ValueError(msg)

    def chunk_documents(
        self,
        documents: list[Document],
        add_metadata: bool = True,
    ) -> list[Document]:
        """Chunk documents into smaller pieces.

        Args:
            documents: List of documents to chunk.
            add_metadata: Whether to add chunk metadata (index, source, etc.).

        Returns:
            List of chunked documents.
        """
        chunks: list[Document] = []

        for doc_idx, doc in enumerate(documents):
            doc_chunks = self.splitter.split_documents([doc])

            for chunk_idx, chunk in enumerate(doc_chunks):
                if add_metadata:
                    chunk.metadata = chunk.metadata.copy()
                    chunk.metadata["chunk_index"] = chunk_idx
                    chunk.metadata["total_chunks"] = len(doc_chunks)
                    chunk.metadata["doc_index"] = doc_idx

                    # Preserve source information
                    if "source" not in chunk.metadata:
                        chunk.metadata["source"] = doc.metadata.get("source", "unknown")

                chunks.append(chunk)

        return chunks

    def chunk_text(self, text: str) -> list[str]:
        """Chunk a single text string.

        Args:
            text: Text to chunk.

        Returns:
            List of text chunks.
        """
        return self.splitter.split_text(text)

