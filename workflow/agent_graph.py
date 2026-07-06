"""LangGraph orchestration for the 红楼梦 multi-agent QA workflow."""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.role_agents import build_deepseek_llm, run_character_agent, run_knowledge_agent
from config import validate_runtime_config
from memory.session_memory import session_memory


Intent = Literal["knowledge", "character"]


class AgentState(TypedDict):
    """Global state shared across the LangGraph workflow."""

    session_id: str
    user_input: str
    chat_history: Annotated[list[BaseMessage], add_messages]
    intent: str
    task_plan: str
    retrieved_context: list[dict[str, Any]]
    intermediate_result: str
    final_answer: str
    sources: list[dict[str, str]]
    error: str


def _extract_sources(text: str) -> list[dict[str, str]]:
    """Extract chapter references from agent text and tool outputs."""

    pattern = re.compile(r"(第[一二三四五六七八九十百零〇\d]+回)\s*([^\n\[]*)")
    seen: set[tuple[str, str]] = set()
    sources: list[dict[str, str]] = []
    for chapter_no, title in pattern.findall(text):
        normalized_title = title.strip(" ：:，,。；;")
        key = (chapter_no, normalized_title)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"chapter_no": chapter_no, "chapter_title": normalized_title})
    return sources[:8]


def _collect_tool_context(intermediate_steps: list[Any]) -> list[dict[str, Any]]:
    """Collect tool observations from AgentExecutor intermediate steps."""

    contexts: list[dict[str, Any]] = []
    for step in intermediate_steps:
        if not isinstance(step, tuple) or len(step) != 2:
            continue
        action, observation = step
        contexts.append(
            {
                "tool": getattr(action, "tool", "unknown_tool"),
                "tool_input": getattr(action, "tool_input", {}),
                "observation": str(observation),
            }
        )
    return contexts


def identify_intent_node(state: AgentState) -> dict[str, Any]:
    """Identify intent and create a lightweight task plan."""

    query = state["user_input"]
    history_hint = "\n".join([msg.content for msg in state.get("chat_history", [])[-4:]])
    prompt = f"""请判断用户关于《红楼梦》的问题意图，并拆解任务。

只能返回JSON，不要添加其他文本：
{{
  "intent": "knowledge 或 character",
  "task_plan": "一句话说明任务拆解"
}}

判定标准：
- knowledge：原文细节、情节、回目、地点、诗词、事件事实。
- character：人物身份、关系、形象、性格、命运结局、人物对比。

近期对话：
{history_hint}

当前问题：{query}
"""
    try:
        llm = build_deepseek_llm(temperature=0)
        response = llm.invoke(prompt)
        data = json.loads(response.content)
        intent = data.get("intent", "knowledge")
        if intent not in {"knowledge", "character"}:
            intent = "knowledge"
        return {"intent": intent, "task_plan": data.get("task_plan", "根据问题路由到对应Agent处理。")}
    except Exception:
        character_keywords = ["人物", "性格", "形象", "关系", "结局", "身份", "命运", "黛玉", "宝玉", "宝钗", "王熙凤"]
        intent = "character" if any(keyword in query for keyword in character_keywords) else "knowledge"
        return {"intent": intent, "task_plan": "LLM意图识别失败，使用关键词规则完成路由。"}


def route_by_intent(state: AgentState) -> str:
    """Conditional edge router."""

    return "character_agent" if state.get("intent") == "character" else "knowledge_agent"


def knowledge_agent_node(state: AgentState) -> dict[str, Any]:
    """Run the knowledge QA specialist agent."""

    try:
        result = run_knowledge_agent(state["user_input"], state.get("chat_history", []))
        output = result.get("output", "")
        contexts = _collect_tool_context(result.get("intermediate_steps", []))
        return {"intermediate_result": output, "retrieved_context": contexts}
    except Exception as exc:
        return {"error": f"知识问答Agent执行失败：{exc}", "intermediate_result": ""}


def character_agent_node(state: AgentState) -> dict[str, Any]:
    """Run the character analysis specialist agent."""

    try:
        result = run_character_agent(state["user_input"], state.get("chat_history", []))
        output = result.get("output", "")
        contexts = _collect_tool_context(result.get("intermediate_steps", []))
        return {"intermediate_result": output, "retrieved_context": contexts}
    except Exception as exc:
        return {"error": f"人物分析Agent执行失败：{exc}", "intermediate_result": ""}


def validation_node(state: AgentState) -> dict[str, Any]:
    """Validate answer boundaries and normalize final output."""

    if state.get("error"):
        return {
            "final_answer": (
                f"{state['error']}\n\n"
                "请确认 .env 配置、DeepSeek API 可用性，以及是否已安装依赖并存在 红楼梦.txt。"
            ),
            "sources": [],
        }

    evidence_text = "\n\n".join(str(item.get("observation", "")) for item in state.get("retrieved_context", []))
    draft_answer = state.get("intermediate_result", "")
    sources = _extract_sources(evidence_text + "\n" + draft_answer)

    prompt = f"""你是《红楼梦》问答系统的答案校验节点。请基于检索证据校验子Agent草稿。

要求：
1. 保留有证据支持的回答。
2. 补充或修正回目标注。
3. 明确回答边界，证据不足处必须说明“不足以确认”。
4. 输出最终中文答案，不要输出校验过程。

用户问题：{state["user_input"]}

子Agent草稿：
{draft_answer}

检索证据：
{evidence_text}
"""
    try:
        llm = build_deepseek_llm(temperature=0)
        response = llm.invoke(prompt)
        final_answer = response.content.strip()
    except Exception:
        final_answer = draft_answer.strip() or "当前未能生成有效答案，请稍后重试。"

    if sources and "依据回目" not in final_answer:
        source_text = "、".join(
            f"{item['chapter_no']} {item.get('chapter_title', '')}".strip() for item in sources[:5]
        )
        final_answer = f"{final_answer}\n\n依据回目：{source_text}"

    return {"final_answer": final_answer, "sources": sources}


def build_agent_graph():
    """Build and compile the LangGraph workflow."""

    graph = StateGraph(AgentState)
    graph.add_node("identify_intent", identify_intent_node)
    graph.add_node("knowledge_agent", knowledge_agent_node)
    graph.add_node("character_agent", character_agent_node)
    graph.add_node("validate_answer", validation_node)

    graph.set_entry_point("identify_intent")
    graph.add_conditional_edges(
        "identify_intent",
        route_by_intent,
        {
            "knowledge_agent": "knowledge_agent",
            "character_agent": "character_agent",
        },
    )
    graph.add_edge("knowledge_agent", "validate_answer")
    graph.add_edge("character_agent", "validate_answer")
    graph.add_edge("validate_answer", END)
    return graph.compile()


_COMPILED_GRAPH = None


def get_compiled_graph():
    """Return a process-level compiled graph."""

    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_agent_graph()
    return _COMPILED_GRAPH


def run_agent_workflow(query: str, session_id: str) -> dict[str, Any]:
    """Run the full multi-agent workflow with session memory."""

    validate_runtime_config()
    history = session_memory.get_history(session_id)
    session_memory.append_user_message(session_id, query)

    initial_state: AgentState = {
        "session_id": session_id,
        "user_input": query,
        "chat_history": history,
        "intent": "",
        "task_plan": "",
        "retrieved_context": [],
        "intermediate_result": "",
        "final_answer": "",
        "sources": [],
        "error": "",
    }

    graph = get_compiled_graph()
    result = graph.invoke(initial_state)
    final_answer = result.get("final_answer", "")
    session_memory.append_ai_message(session_id, final_answer)
    return result
