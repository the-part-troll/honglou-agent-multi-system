"""LangChain tools used by the specialist agents."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config import Config
from knowledge_base import get_knowledge_base


class TextSearchInput(BaseModel):
    """Input schema for source text retrieval."""

    query: str = Field(..., description="用于检索《红楼梦》原文片段的查询词或问题")


class CharacterSearchInput(BaseModel):
    """Input schema for character information retrieval."""

    name: str = Field(..., description="需要查询的人物姓名，例如：林黛玉、贾宝玉、王熙凤")


def _format_results(results: list[dict[str, Any]]) -> str:
    """Format retrieval results into a compact evidence block for LLM agents."""

    if not results:
        return "未检索到相关原文片段。"

    formatted: list[str] = []
    for idx, item in enumerate(results, start=1):
        metadata = item.get("metadata", {})
        chapter_no = metadata.get("chapter_no", "未知回目")
        chapter_title = metadata.get("chapter_title", "")
        content = item.get("content", "").replace("\n", " ").strip()
        formatted.append(
            f"[{idx}] {chapter_no} {chapter_title}\n"
            f"相关度分数: {item.get('score', 0):.4f}\n"
            f"原文片段: {content}"
        )
    return "\n\n".join(formatted)


def retrieve_original_text(query: str) -> str:
    """Retrieve top-k relevant original text chunks."""

    kb = get_knowledge_base()
    results = kb.search(query=query, top_k=Config.RETRIEVAL_TOP_K)
    return _format_results(results)


def retrieve_character_info(name: str) -> str:
    """Retrieve key passages about a character."""

    kb = get_knowledge_base()
    results = kb.search_character(name=name, top_k=Config.RETRIEVAL_TOP_K)
    return _format_results(results)


original_text_retrieval_tool = StructuredTool.from_function(
    func=retrieve_original_text,
    name="original_text_retrieval",
    description=(
        "检索《红楼梦》原文。适用于回目、情节、诗词、地点、事件、原文细节等事实问题。"
        "输入应是简洁查询词或完整问题，返回Top-5原文片段及回目元数据。"
    ),
    args_schema=TextSearchInput,
)

character_info_tool = StructuredTool.from_function(
    func=retrieve_character_info,
    name="character_info_query",
    description=(
        "查询《红楼梦》人物信息。适用于人物身份、关系、性格、命运结局、关键出场段落等问题。"
        "输入应是单个人名，返回Top-5关键原文片段及回目元数据。"
    ),
    args_schema=CharacterSearchInput,
)


def get_retrieval_tools() -> list[StructuredTool]:
    """Return all project tools."""

    return [original_text_retrieval_tool, character_info_tool]
