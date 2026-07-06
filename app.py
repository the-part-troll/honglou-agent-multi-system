"""Streamlit chat UI for the 红楼梦 multi-agent QA system."""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from knowledge_base import get_knowledge_base
from memory.session_memory import session_memory
from workflow.agent_graph import run_agent_workflow


st.set_page_config(page_title="红楼梦知识多Agent问答系统", page_icon="📚", layout="wide")


def _init_state() -> None:
    """Initialize Streamlit session state."""

    if "sessions" not in st.session_state:
        first_id = str(uuid.uuid4())
        st.session_state.sessions = {first_id: {"title": "新会话", "messages": []}}
        st.session_state.current_session_id = first_id


def _current_session() -> dict[str, Any]:
    """Return current UI session data."""

    return st.session_state.sessions[st.session_state.current_session_id]


def _new_session() -> None:
    """Create and switch to a new chat session."""

    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {"title": "新会话", "messages": []}
    st.session_state.current_session_id = session_id


def _render_sources(result: dict[str, Any]) -> None:
    """Render source and tool evidence in expandable panels."""

    sources = result.get("sources", [])
    contexts = result.get("retrieved_context", [])
    if sources:
        with st.expander("引用回目"):
            for source in sources:
                chapter_no = source.get("chapter_no", "未知回目")
                title = source.get("chapter_title", "")
                st.write(f"- {chapter_no} {title}".strip())
    if contexts:
        with st.expander("工具调用与原文证据"):
            for idx, item in enumerate(contexts, start=1):
                st.markdown(f"**工具 {idx}: `{item.get('tool', 'unknown')}`**")
                st.code(str(item.get("observation", ""))[:3000], language="text")


_init_state()

with st.sidebar:
    st.title("红楼梦多Agent")
    if st.button("新建会话", use_container_width=True):
        _new_session()
        st.rerun()

    session_ids = list(st.session_state.sessions.keys())
    selected_index = session_ids.index(st.session_state.current_session_id)
    selected_session_id = st.radio(
        "会话列表",
        session_ids,
        index=selected_index,
        format_func=lambda sid: st.session_state.sessions[sid]["title"],
    )
    st.session_state.current_session_id = selected_session_id

    if st.button("清空当前会话", use_container_width=True):
        sid = st.session_state.current_session_id
        st.session_state.sessions[sid]["messages"] = []
        session_memory.clear(sid)
        st.rerun()

    st.divider()
    if st.button("初始化/检查知识库", use_container_width=True):
        with st.spinner("正在加载或构建 ChromaDB 向量库..."):
            kb = get_knowledge_base()
            kb.load_or_build()
        st.success("知识库已就绪")

st.title("红楼梦知识多Agent问答系统")
st.caption("LangChain + LangGraph · 总控调度 · 双子Agent · 工具调用 · 会话记忆")

current = _current_session()
for message in current["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("result"):
            _render_sources(message["result"])

query = st.chat_input("请输入关于《红楼梦》的问题，例如：林黛玉初进贾府发生了什么？")
if query:
    if current["title"] == "新会话":
        current["title"] = query[:16]

    current["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("总控调度Agent正在拆解任务并调用子Agent..."):
            try:
                result = run_agent_workflow(query=query, session_id=st.session_state.current_session_id)
                answer = result.get("final_answer", "未生成答案。")
            except Exception as exc:
                result = {"sources": [], "retrieved_context": [], "error": str(exc)}
                answer = f"系统运行失败：{exc}"
        st.markdown(answer)
        _render_sources(result)

    current["messages"].append({"role": "assistant", "content": answer, "result": result})
