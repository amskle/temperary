"""LangGraph agent definition for LabReportAgent — ReAct-style pipeline."""

import json
import re
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from config import config
from memory import init_memory, retrieve_similar
from tools import (
    analyze_code,
    execute_code_sandbox,
    fill_template,
    generate_report,
    generate_report_iterative,
    parse_template,
)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(dict):
    """Mutable state dict used as LangGraph state."""

    template_path: str = ""
    user_requirement: str = ""
    code: str = ""
    code_analysis: str = ""
    code_execution_output: str = ""
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

    # Build template headers string for better embedding accuracy
    template_headers = " | ".join(
        f["field_id"] for f in fields
    ) if fields else ""

    # Retrieve similar cases — query embeds both template structure and requirement
    examples = retrieve_similar(
        state["user_requirement"],
        template_headers=template_headers,
    )
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
    """If code is provided, analyse it with the LLM AND execute it for real output."""
    if not state.get("code", "").strip():
        print("\n" + "=" * 60)
        print("  STEP 2: Code Analysis & Execution (skipped)")
        print("=" * 60)
        print("  Thought: No source code provided — skip analysis.")
        print("  Action: None")
        return {"code_analysis": "", "code_execution_output": ""}

    code = state["code"]
    print("\n" + "=" * 60)
    print("  STEP 2: Analyze & Execute Source Code")
    print("=" * 60)
    print("  Thought: The user provided source code. I will statically "
          "analyze it AND execute it to capture real output.")
    print("  Action: tools.analyze_code() → LLM")
    print("  Action: tools.execute_code_sandbox() → subprocess")

    # Static analysis via LLM
    analysis = analyze_code(code)
    print(f"  Observation: Code analysis generated ({len(analysis)} chars).")

    # Dynamic execution for ground-truth output
    exec_result = execute_code_sandbox(code)
    output_parts = []
    if exec_result.get("stdout"):
        output_parts.append(f"stdout:\n{exec_result['stdout']}")
    if exec_result.get("stderr"):
        output_parts.append(f"stderr:\n{exec_result['stderr']}")
    if exec_result.get("error"):
        output_parts.append(f"execution error: {exec_result['error']}")
    output_parts.append(f"exit code: {exec_result.get('exit_code', '-1')}")

    exec_output = "\n".join(output_parts)
    print(f"  Observation: Code execution completed "
          f"(exit={exec_result.get('exit_code')}, "
          f"stdout={len(exec_result.get('stdout', ''))} chars, "
          f"stderr={len(exec_result.get('stderr', ''))} chars).")

    return {
        "code_analysis": analysis,
        "code_execution_output": exec_output,
    }


def generate_content(state: AgentState) -> dict[str, Any]:
    """Generate content for all variable fields iteratively via DeepSeek."""
    retry = state.get("generation_retry_count", 0)
    print("\n" + "=" * 60)
    print(f"  STEP 3: Generate Report Content (attempt {retry + 1})")
    print("=" * 60)

    has_few_shot = bool(state.get("few_shot"))
    has_code_exec = bool(state.get("code_execution_output", "").strip())
    print("  Thought: I have the template structure, user requirements, "
          f"{'and ' + str(len(state.get('few_shot', []))) + ' few-shot example(s)' if has_few_shot else 'no few-shot examples'}."
          f"{' Real code execution output is available.' if has_code_exec else ''}"
          " Now I will generate content for each variable field iteratively.")
    print("  Action: tools.generate_report_iterative() → LLM (multi-pass)")

    content = generate_report_iterative(
        report_requirements=state["user_requirement"],
        code_analysis=state.get("code_analysis", ""),
        code_execution_output=state.get("code_execution_output", ""),
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


# ---------------------------------------------------------------------------
# Validation heuristics
# ---------------------------------------------------------------------------

# Patterns that indicate placeholder/fallback content rather than real prose
_PLACEHOLDER_PATTERNS = [
    r"\[.*?(?:Content for|Generated content|see generated|placeholder|TODO|TBD|insert|to be filled).*?\]",
    r"\{\{.*?\}\}",
    r"<.*?(?:field|placeholder|insert|todo).*?>",
]
_PLACEHOLDER_KEYWORDS = [
    "see generated output",
    "placeholder",
    "[content for",
    "[generated content for",
    "lorem ipsum",
]


def _is_cjk(text: str) -> bool:
    """Return True if text contains significant CJK characters (>10%)."""
    cjk_count = sum(
        1 for ch in text
        if ("一" <= ch <= "鿿")       # CJK Unified Ideographs
        or ("㐀" <= ch <= "䶿")       # CJK Extension A
        or ("　" <= ch <= "〿")       # CJK Symbols/Punctuation
        or ("぀" <= ch <= "ヿ")       # Hiragana/Katakana
    )
    return cjk_count > len(text) * 0.1


def _looks_like_placeholder(text: str, is_complex_field: bool = True) -> bool:
    """Check if text looks like a placeholder/fallback rather than real content.

    For simple metadata fields (name, student_id, date), only checks for
    explicit placeholder artifacts. For complex academic fields, also checks
    word/character count and sentence structure.
    """
    text_lower = text.lower().strip()
    # Pattern match
    for pattern in _PLACEHOLDER_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    # Keyword match
    for keyword in _PLACEHOLDER_KEYWORDS:
        if keyword in text_lower:
            return True

    # For simple fields, being short is expected — skip content-length checks
    if not is_complex_field:
        return False

    # CJK text: check character count instead of word count
    if _is_cjk(text):
        # Strip whitespace/punctuation for char count
        chars = re.sub(r"\s+", "", text)
        if len(chars) < config.min_word_count * 3:
            return True
        # Check for at least 2 CJK sentence delimiters (。！？)
        sentence_marks = sum(1 for ch in text if ch in "。！？\n")
        if sentence_marks < 1:
            return True
    else:
        # Latin text: check word count
        words = text_lower.split()
        if len(words) < config.min_word_count:
            return True
        # Check sentence count
        sentences = re.split(r"[.!?]\s+", text)
        real_sentences = [s for s in sentences if len(s.split()) >= 5]
        if len(real_sentences) < 2:
            return True

    return False


def _llm_quality_check(content: dict[str, str], fields: list[dict]) -> list[str]:
    """Use a lightweight LLM call to judge whether content is legitimate.

    Returns a list of field_ids that FAIL the quality check.
    """
    if not config.enable_llm_judge:
        return []

    from tools import _call_llm

    # Build a compact evaluation prompt with all fields
    fields_block = "\n\n".join(
        f"[{f['field_id']}]: {content.get(f['field_id'], '')[:300]}"
        for f in fields
    )
    prompt = (
        "For each field below, rate whether the content reads like "
        "legitimate academic lab report prose (1-5):\n"
        "5 = detailed, substantive academic content\n"
        "1 = placeholder, generic, or fallback text\n\n"
        f"{fields_block}\n\n"
        "Return ONLY a JSON object mapping field_id to rating. "
        'Example: {"purpose": 4, "result": 1}'
    )

    try:
        raw = _call_llm(
            system_prompt="You are a strict academic content evaluator. "
            "Return ONLY valid JSON.",
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=512,
            step_name="llm_judge",
        )
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        ratings = json.loads(json_match.group(0)) if json_match else {}
    except Exception:
        return []

    # Fields rated 1-2 are considered failures
    failed = [fid for fid, rating in ratings.items() if isinstance(rating, (int, float)) and rating < 3]
    if failed:
        print(f"  LLM Judge: flagged fields {failed} as insufficient quality.")
    return failed


# Fields that need deep academic content (same as tools._COMPLEX_FIELD_IDS)
_COMPLEX_FIELD_IDS = {"purpose", "principle", "steps", "result", "analysis", "conclusion"}


def validate_content(state: AgentState) -> Literal["generate_content", "fill_and_save"]:
    """Validate generated content; retry if placeholder-like, empty, or too short."""
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
        is_complex = fid in _COMPLEX_FIELD_IDS

        # Check 1: placeholder artifacts
        if _looks_like_placeholder(val, is_complex_field=is_complex):
            preview = val[:120].replace("\n", " ")
            print(
                f"  Field '{fid}' looks like placeholder/fallback. "
                f"Preview: '{preview}…'"
            )
            all_ok = False
            continue

        # Check 2: bare minimum length
        min_len = 2 if not is_complex else 10
        if len(val) < min_len:
            print(f"  Field '{fid}' too short ({len(val)} chars).")
            all_ok = False
            continue

        if is_complex:
            chars = len(re.sub(r"\s+", "", val))
            print(f"  Field '{fid}': OK ({len(val)} chars, {chars} content chars).")
        else:
            print(f"  Field '{fid}': OK ({len(val)} chars, simple field).")

    # Check 3: LLM-as-judge (complex fields only)
    if all_ok and config.enable_llm_judge:
        complex_fields = [f for f in fields if f["field_id"] in _COMPLEX_FIELD_IDS]
        if complex_fields:
            failed_by_judge = _llm_quality_check(content, complex_fields)
            if failed_by_judge:
                all_ok = False

    if not all_ok and retry_count <= config.max_generation_retries:
        print(
            f"  Thought: Some fields have insufficient quality. "
            f"Retry (attempt {retry_count}/{config.max_generation_retries})."
        )
        return "generate_content"

    if all_ok:
        print("  Observation: All fields pass validation (heuristics + LLM judge).")
    else:
        print("  Observation: Proceeding despite issues (max retries reached).")
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
