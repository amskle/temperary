"""LangGraph agent definition for LabReportAgent — ReAct-style pipeline."""

import json
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from config import config
from memory import init_memory, retrieve_similar
from tools import analyze_code, fill_template, generate_report, parse_template

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(dict):
    """Mutable state dict used as LangGraph state."""

    template_path: str = ""
    user_requirement: str = ""
    code: str = ""
    code_analysis: str = ""
    few_shot: list[dict] = []
    template_info: dict[str, Any] = {}
    filled_content: dict[str, str] = {}
    output_path: str = ""
    errors: list[str] = []
    generation_retry_count: int = 0


# ---------------------------------------------------------------------------
# Node functions — each node prints a ReAct-style Thought/Action/Observation
# ---------------------------------------------------------------------------


def parse_and_retrieve(state: AgentState) -> dict[str, Any]:
    """Parse the template and retrieve similar historical cases."""
    print("\n" + "=" * 60)
    print("  STEP 1: Parse Template & Retrieve Memory")
    print("=" * 60)
    print("  Thought: I need to read the .docx template and identify "
          "immutable sections vs. variable fields.")
    print("  Action: parse_template() + memory.retrieve_similar()")

    # Parse template
    info = parse_template(state["template_path"])
    fields = info.get("variable_fields", [])
    immutable_count = len(info["immutable_sections"])
    print(f"  Observation: Found {immutable_count} immutable section(s), "
          f"{len(fields)} variable field(s)")
    if fields:
        for f in fields:
            print(f"    - {f['field_id']}  (placeholder: {f['placeholder']})")

    # Initialize memory
    init_memory()
    print("  Observation: Vector memory initialised.")

    # Retrieve similar cases
    examples = retrieve_similar(state["user_requirement"])
    if examples:
        print(f"  Observation: Found {len(examples)} similar historical case(s).")
        for ex in examples:
            print(f"    - [{ex['id']}] {ex['requirement'][:100]}")
    else:
        print("  Observation: No similar historical cases found.")

    return {
        "template_info": info,
        "few_shot": examples,
        "generation_retry_count": 0,
    }


def analyze_code_if_needed(state: AgentState) -> dict[str, Any]:
    """If code is provided, analyse it with the LLM."""
    if not state.get("code", "").strip():
        print("\n" + "=" * 60)
        print("  STEP 2: Code Analysis (skipped)")
        print("=" * 60)
        print("  Thought: No source code provided — skip analysis.")
        print("  Action: None")
        return {"code_analysis": ""}

    print("\n" + "=" * 60)
    print("  STEP 2: Analyze Source Code")
    print("=" * 60)
    print("  Thought: The user provided source code. I need to understand "
          "its purpose and expected output.")
    print("  Action: tools.analyze_code() → LLM")

    analysis = analyze_code(state["code"])
    print(f"  Observation: Code analysis generated ({len(analysis)} chars).")
    return {"code_analysis": analysis}


def generate_content(state: AgentState) -> dict[str, Any]:
    """Generate content for all variable fields via DeepSeek."""
    retry = state.get("generation_retry_count", 0)
    print("\n" + "=" * 60)
    print(f"  STEP 3: Generate Report Content (attempt {retry + 1})")
    print("=" * 60)

    has_few_shot = bool(state.get("few_shot"))
    print("  Thought: I have the template structure, user requirements, "
          f"{'and ' + str(len(state.get('few_shot', []))) + ' few-shot example(s)' if has_few_shot else 'no few-shot examples'}."
          " Now I will generate content for each variable field.")
    print("  Action: tools.generate_report() → LLM")

    content = generate_report(
        report_requirements=state["user_requirement"],
        code_analysis=state.get("code_analysis", ""),
        template_info=state["template_info"],
        few_shot_examples=state.get("few_shot"),
    )

    print(f"  Observation: Generated content for {len(content)} field(s).")
    for k, v in content.items():
        preview = v[:100].replace("\n", " ")
        print(f"    {k}: {preview}…")

    return {
        "filled_content": content,
        "generation_retry_count": retry + 1,
    }


def validate_content(state: AgentState) -> Literal["generate_content", "fill_and_save"]:
    """Validate generated content; retry if empty or too short."""
    content = state.get("filled_content", {})
    fields = state["template_info"].get("variable_fields", [])
    retry_count = state.get("generation_retry_count", 0)

    print("\n" + "-" * 40)
    print("  [Validation Gate]")
    print("-" * 40)

    if not content:
        print("  Thought: No content was generated. This is likely a failure.")
        if retry_count <= config.max_generation_retries:
            print(f"  Action: Retry generation (attempt {retry_count}/{config.max_generation_retries})")
            return "generate_content"
        print("  Action: Max retries reached — proceed with empty content.")
        return "fill_and_save"

    all_ok = True
    for field in fields:
        fid = field["field_id"]
        val = content.get(fid, "").strip()
        if len(val) < 10:
            print(
                f"  Field '{fid}' too short ({len(val)} chars). "
                f"Content: '{val}'"
            )
            all_ok = False

    if not all_ok and retry_count <= config.max_generation_retries:
        print(
            f"  Thought: Some fields have insufficient content. "
            f"Retry (attempt {retry_count}/{config.max_generation_retries})."
        )
        return "generate_content"

    if all_ok:
        print("  Observation: All fields pass validation.")
    else:
        print("  Observation: Proceeding despite short fields (max retries reached).")
    return "fill_and_save"


def fill_and_save(state: AgentState) -> dict[str, Any]:
    """Fill the template with generated content and save."""
    print("\n" + "=" * 60)
    print("  STEP 4: Fill Template & Save")
    print("=" * 60)
    print("  Thought: All content is ready. I will now replace placeholders "
          "in the template and save the output.")
    print("  Action: tools.fill_template()")

    path = fill_template(
        template_path=state["template_path"],
        content_mapping=state["filled_content"],
    )
    print(f"  Observation: Report saved to: {path}")

    return {"output_path": path}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def build_agent() -> CompiledStateGraph:
    """Construct the LangGraph state machine."""
    builder = StateGraph(AgentState)

    builder.add_node("parse_and_retrieve", parse_and_retrieve)
    builder.add_node("analyze_code_if_needed", analyze_code_if_needed)
    builder.add_node("generate_content", generate_content)
    builder.add_node("fill_and_save", fill_and_save)

    builder.set_entry_point("parse_and_retrieve")

    builder.add_edge("parse_and_retrieve", "analyze_code_if_needed")
    builder.add_edge("analyze_code_if_needed", "generate_content")
    builder.add_conditional_edges(
        "generate_content",
        validate_content,
        {
            "generate_content": "generate_content",
            "fill_and_save": "fill_and_save",
        },
    )
    builder.add_edge("fill_and_save", END)

    return builder.compile()


def run_agent(
    template_path: str,
    user_requirement: str,
    code: str = "",
) -> str:
    """Convenience wrapper: build graph, run, return output path."""
    agent = build_agent()

    initial: AgentState = AgentState(
        template_path=template_path,
        user_requirement=user_requirement,
        code=code,
    )

    result = agent.invoke(initial)
    return result.get("output_path", "")
