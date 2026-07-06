"""Minimal command-line test for the multi-agent workflow."""

from __future__ import annotations

from pprint import pprint

from workflow.agent_graph import run_agent_workflow


def main() -> None:
    """Run two sample questions to validate routing, tools, and memory."""

    session_id = "cli-test-session"
    questions = [
        "林黛玉初进贾府主要发生了什么？请标注依据回目。",
        "她和贾宝玉的关系有什么特点？",
    ]

    for question in questions:
        print("=" * 80)
        print(f"Q: {question}")
        result = run_agent_workflow(query=question, session_id=session_id)
        print("\nA:")
        print(result.get("final_answer", ""))
        print("\nSources:")
        pprint(result.get("sources", []))


if __name__ == "__main__":
    main()
