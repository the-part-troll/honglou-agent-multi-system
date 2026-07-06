"""Knowledge base construction and retrieval for 红楼梦."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


@dataclass(frozen=True)
class Chapter:
    """A parsed chapter from the novel."""

    chapter_no: str
    title: str
    text: str


class HongLouMengKnowledgeBase:
    """Build, load, and query a ChromaDB vector store for 红楼梦."""

    def __init__(self) -> None:
        self.novel_path = Config.NOVEL_PATH
        self.chroma_dir = Config.CHROMA_DIR
        self.collection_name = Config.CHROMA_COLLECTION_NAME
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
        self.vectorstore: Chroma | None = None

    def load_or_build(self) -> Chroma:
        """Load a persisted Chroma store, or build it from the novel text."""

        if self.vectorstore is not None:
            return self.vectorstore

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.chroma_dir),
        )

        if vectorstore._collection.count() == 0:
            documents = self.build_documents()
            if not documents:
                raise RuntimeError("No documents were built from 红楼梦.txt")
            vectorstore.add_documents(documents)

        self.vectorstore = vectorstore
        return vectorstore

    def build_documents(self) -> list[Document]:
        """Parse chapters and split them into overlapping text chunks."""

        chapters = self.parse_chapters(self.novel_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )

        documents: list[Document] = []
        for chapter in chapters:
            chunks = splitter.split_text(chapter.text)
            for idx, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "chapter_no": chapter.chapter_no,
                            "chapter_title": chapter.title,
                            "chunk_id": idx,
                            "source": str(self.novel_path.name),
                        },
                    )
                )
        return documents

    @staticmethod
    def parse_chapters(path: Path) -> list[Chapter]:
        """Read UTF-8 text and split chapters by headings like '第X回'."""

        if not path.exists():
            raise FileNotFoundError(f"Novel file not found: {path}")

        text = path.read_text(encoding="utf-8")
        pattern = re.compile(r"(?m)^(第[一二三四五六七八九十百零〇\d]+回)\s*(.*)$")
        matches = list(pattern.finditer(text))
        if not matches:
            raise ValueError("No chapter headings matched. Expected headings beginning with '第X回'.")

        chapters: list[Chapter] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            chapter_no = match.group(1).strip()
            title = match.group(2).strip() or chapter_no
            chapter_text = text[start:end].strip()
            chapters.append(Chapter(chapter_no=chapter_no, title=title, text=chapter_text))

        return chapters

    def search(self, query: str, top_k: int = Config.RETRIEVAL_TOP_K) -> list[dict[str, Any]]:
        """Return top-k semantically relevant chunks with scores and metadata."""

        vectorstore = self.load_or_build()
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]

    def search_character(self, name: str, top_k: int = Config.RETRIEVAL_TOP_K) -> list[dict[str, Any]]:
        """Retrieve character-related passages using a focused query."""

        query = f"{name} 人物 身份 关系 结局 性格 出场"
        return self.search(query=query, top_k=top_k)


_KB: HongLouMengKnowledgeBase | None = None


def get_knowledge_base() -> HongLouMengKnowledgeBase:
    """Return a process-level knowledge base singleton."""

    global _KB
    if _KB is None:
        _KB = HongLouMengKnowledgeBase()
    return _KB
