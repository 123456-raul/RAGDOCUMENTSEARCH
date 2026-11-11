"""Saves document chunks and their embeddings to disk.

Loading and embedding the sources is the slow part of startup. With a saved
copy, a restart only has to read two files, as long as the sources and the
chunking settings are unchanged.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document


class IndexCache:
    """Reads and writes the saved chunks and embeddings"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.meta_file = self.cache_dir / "meta.json"
        self.chunks_file = self.cache_dir / "chunks.json"
        self.vectors_file = self.cache_dir / "vectors.npy"

    @staticmethod
    def make_key(sources: List[str], **settings) -> str:
        """
        Fingerprint of everything the index depends on.

        Covers the source list, the content of local files, and the given
        settings (chunk size, embedding model). A changed web page does not
        change the key, so rebuild the index to pick up new page content.
        """
        parts = [json.dumps(settings, sort_keys=True)]
        for src in sources:
            if src.startswith(("http://", "https://")):
                parts.append(src)
                continue

            path = Path(src)
            files = sorted(path.glob("**/*.pdf")) if path.is_dir() else [path]
            for file in files:
                if file.is_file():
                    digest = hashlib.sha256(file.read_bytes()).hexdigest()
                    parts.append(f"{file.name}:{digest}")
                else:
                    parts.append(f"{file.name}:missing")

        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def load(self, key: str) -> Optional[Tuple[List[Document], np.ndarray]]:
        """Return the saved chunks and vectors, or None if they can't be used"""
        try:
            meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
            if meta.get("key") != key:
                return None

            chunks = json.loads(self.chunks_file.read_text(encoding="utf-8"))
            vectors = np.load(self.vectors_file)
        except (OSError, ValueError):
            # Missing or damaged files: fall back to building the index
            return None

        if len(chunks) == 0 or len(chunks) != len(vectors):
            return None

        documents = [
            Document(page_content=c["page_content"], metadata=c["metadata"])
            for c in chunks
        ]
        return documents, vectors

    def save(self, key: str, documents: List[Document], vectors: np.ndarray) -> None:
        """Write the chunks and vectors to disk"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Remove the old meta file first, so a half-written save is never loaded
        self.clear()

        chunks = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        self.chunks_file.write_text(
            json.dumps(chunks, ensure_ascii=False, default=str), encoding="utf-8"
        )
        np.save(self.vectors_file, np.asarray(vectors, dtype=np.float32))

        meta = {
            "key": key,
            "num_chunks": len(documents),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def clear(self) -> None:
        """Forget the saved index, so the next startup builds it again"""
        self.meta_file.unlink(missing_ok=True)
