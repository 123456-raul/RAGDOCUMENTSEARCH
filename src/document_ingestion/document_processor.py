"""Document processing module for loading and splitting documents"""

from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT_SUFFIXES = (".txt", ".md")


def source_label(doc: Document) -> str:
    """
    Short description of where a chunk came from, for showing next to an answer.

    Web pages give their title, PDFs their file name and page number.
    """
    meta = doc.metadata or {}
    source = str(meta.get("source") or "")

    if source.startswith(("http://", "https://")):
        title = str(meta.get("title") or "").strip()
        return title or urlparse(source).netloc or source

    name = Path(source).name if source else "Unknown source"
    # pypdf counts pages from 0; page_label is the number printed on the page
    page = meta.get("page_label")
    if page is None and meta.get("page") is not None:
        page = int(meta["page"]) + 1
    return f"{name}, page {page}" if page is not None else name


def source_url(doc: Document) -> str:
    """The chunk's web address, or an empty string for local files"""
    source = str((doc.metadata or {}).get("source") or "")
    return source if source.startswith(("http://", "https://")) else ""


class DocumentProcessor:
    """Handles document loading and processing"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        """
        Initialize document processor

        Args:
            chunk_size: Size of text chunks, in characters
            chunk_overlap: Characters shared by neighbouring chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load document(s) from a URL"""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a text or Markdown file"""
        loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load document(s) from a PDF file"""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents from URLs, PDF directories, PDF files or text files

        Args:
            sources: List of URLs, PDF folder paths, or file paths

        Returns:
            List of loaded documents
        """
        docs: List[Document] = []
        for src in sources:
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
                continue

            path = Path(src)
            if path.is_dir():  # PDF directory
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() in TEXT_SUFFIXES:
                docs.extend(self.load_from_txt(path))
            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))
            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use a URL, a PDF directory, or a .pdf, .txt or .md file."
                )
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)

    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents

        Args:
            urls: List of sources to process

        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)


if __name__ == "__main__":
    processor = DocumentProcessor()
    chunks = processor.process_urls(["https://example.com"])
    print(f"Loaded {len(chunks)} chunks")
    print(chunks[0].page_content[:200])
