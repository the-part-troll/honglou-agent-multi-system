"""Specialist LangChain agents for the multi-agent QA system.

This module intentionally avoids importing ``langchain.agents``. In some
Windows conda environments that import path eagerly loads aiohttp and the
system certificate store, which can fail before our application starts. The
implementation below still uses LangChain tool binding and an agent-style
tool-calling loop, but keeps the import surface small and stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from config import Config
from tools.retrieval_tools import character_info_tool, original_text_retrieval_tool


@dataclass
class ToolAction:
    """Small compatibility object for workflow intermediate-step collection."""

    tool: str
    tool_input: dict[str, Any]


def build_deepseek_llm(temperature: float | None = None) -> ChatOpenAI:
    """Create a DeepSeek chat model through the OpenAI-compatible client."""

    return ChatOpenAI(
        model=Config.DEEPSEEK_MODEL_NAME,
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL,
        temperature=Config.LLM_TEMPERATURE if temperature is None else temperature,
    )


KNOWLEDGE_AGENT_SYSTEM_PROMPT = """你是《红楼梦》知识问答Agent，专注处理原文细节、情节、回目、地点、诗词、事件事实类问题。

工作规则：
1. 必须优先调用 original_text_retrieval 工具获取原文依据，再作答。
2. 只根据工具返回的证据回答，不要编造未检索到的内容。
3. 回答要标注依据回目，例如“据第三回……”。如果证据不足，明确说明“当前检索证据不足以确认”。
4. 不负责展开复杂人物心理分析；若问题明显是人物形象或关系分析，只回答与原文事实相关的部分。
5. 输出中文，结构清晰、简洁可信。
"""


CHARACTER_AGENT_SYSTEM_PROMPT = """你是《红楼梦》人物分析Agent，专注处理人物身份、关系、形象、性格、命运结局与人物对比类问题。

工作规则：
1. 必须优先调用 character_info_query 工具查询人物相关段落；必要时再调用 original_text_retrieval 补充具体事件证据。
2. 只在证据支持范围内分析，不要把民间说法、电视剧改编或无依据推断当成原著事实。
3. 回答要区分“原文直接依据”和“基于依据的分析判断”。
4. 对人物结局、隐含关系等高风险问题，需要提示版本与证据边界。
5. 输出中文，适合演示AI Agent的可解释推理过程。
"""


def _invoke_tool(tool: BaseTool, tool_args: dict[str, Any]) -> str:
    """Invoke a LangChain tool and normalize the observation to text."""

    try:
        result = tool.invoke(tool_args)
    except TypeError:
        result = tool.invoke(next(iter(tool_args.values())) if tool_args else "")
    return str(result)


def _run_tool_calling_agent(
    *,
    query: str,
    chat_history: list[BaseMessage],
    system_prompt: str,
    tools: list[BaseTool],
) -> dict[str, Any]:
    """Run a compact LangChain tool-calling loop.

    The LLM decides whether to call tools. Tool observations are fed back into
    the model until it returns a final answer or the iteration limit is reached.
    """

    llm = build_deepseek_llm().bind_tools(tools)
    tool_map = {tool.name: tool for tool in tools}
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *chat_history,
        HumanMessage(content=query),
    ]
    intermediate_steps: list[tuple[ToolAction, str]] = []

    for _ in range(Config.MAX_AGENT_ITERATIONS):
        ai_message = llm.invoke(messages)
        messages.append(ai_message)

        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if not tool_calls:
            return {
                "output": str(ai_message.content),
                "intermediate_steps": intermediate_steps,
            }

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {}) or {}
            tool_call_id = tool_call.get("id", "")
            tool = tool_map.get(tool_name)

            if tool is None:
                observation = f"工具 {tool_name} 不存在，无法调用。"
            else:
                observation = _invoke_tool(tool, tool_args)

            intermediate_steps.append((ToolAction(tool=tool_name, tool_input=tool_args), observation))
            messages.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call_id,
                )
            )

    final_prompt = HumanMessage(content="请基于以上工具返回的证据给出最终答案，并明确标注依据回目和证据边界。")
    messages.append(final_prompt)
    final_message = llm.invoke(messages)
    output = final_message.content if isinstance(final_message, AIMessage) else str(final_message)
    return {"output": str(output), "intermediate_steps": intermediate_steps}


def run_knowledge_agent(query: str, chat_history: list[BaseMessage]) -> dict[str, Any]:
    """Run the knowledge QA agent."""

    return _run_tool_calling_agent(
        query=query,
        chat_history=chat_history,
        system_prompt=KNOWLEDGE_AGENT_SYSTEM_PROMPT,
        tools=[original_text_retrieval_tool],
    )


def run_character_agent(query: str, chat_history: list[BaseMessage]) -> dict[str, Any]:
    """Run the character analysis agent."""

    return _run_tool_calling_agent(
        query=query,
        chat_history=chat_history,
        system_prompt=CHARACTER_AGENT_SYSTEM_PROMPT,
        tools=[character_info_tool, original_text_retrieval_tool],
    )
