"""Tools: template parsing, filling, code analysis, report generation."""

import json
import os
import re
import time
from datetime import datetime
from typing import Any

from docx import Document

from config import config

# ---------------------------------------------------------------------------
# Template parsing — detects both {{var}} and <var> style placeholders
# ---------------------------------------------------------------------------

# Match {{word}} or <word> where word contains only word chars
_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}|<(\w+)>")

# Placeholder styles to remember for each field
_PLACEHOLDER_TEMPLATES: dict[str, str] = {}


def _extract_placeholders(text: str) -> list[dict]:
    """Extract placeholder info from text, tracking the original style."""
    results: list[dict] = []
    for match in _VARIABLE_PATTERN.finditer(text):
        # match.group(1) = {{...}}, match.group(2) = <...>
        field_id = match.group(1) or match.group(2)
        raw = match.group(0)
        results.append({"field_id": field_id, "raw_placeholder": raw})
        # Remember which style this field uses (first occurrence wins)
        if field_id not in _PLACEHOLDER_TEMPLATES:
            _PLACEHOLDER_TEMPLATES[field_id] = raw
    return results


def parse_template(template_path: str) -> dict[str, Any]:
    """Parse a .docx template, returning immutable sections and variable fields.

    Returns
        {
          "immutable_sections": [{"type": str, "index": int, "text": str}, ...],
          "variable_fields": [{"field_id": str, "placeholder": str,
                               "context": str}, ...]
        }

    Supports both ``{{field}}`` and ``<field>`` placeholders.
    """
    global _PLACEHOLDER_TEMPLATES
    _PLACEHOLDER_TEMPLATES = {}  # reset per parse

    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    doc = Document(template_path)
    immutable_sections: list[dict[str, Any]] = []
    variable_fields: dict[str, dict[str, Any]] = {}

    # --- Paragraphs ---
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue

        placeholders = _extract_placeholders(text)
        if placeholders:
            for ph in placeholders:
                fid = ph["field_id"]
                if fid not in variable_fields:
                    variable_fields[fid] = {
                        "field_id": fid,
                        "placeholder": ph["raw_placeholder"],
                        "context": text,
                    }

        immutable_sections.append({"type": "paragraph", "index": i, "text": text})

    # --- Tables ---
    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                text = cell.text.strip()
                if not text:
                    continue

                placeholders = _extract_placeholders(text)
                if placeholders:
                    for ph in placeholders:
                        fid = ph["field_id"]
                        if fid not in variable_fields:
                            variable_fields[fid] = {
                                "field_id": fid,
                                "placeholder": ph["raw_placeholder"],
                                "context": (
                                    f"[table {ti}, row {ri}, col {ci}] {text}"
                                ),
                            }

                immutable_sections.append(
                    {
                        "type": "table_cell",
                        "table_index": ti,
                        "row": ri,
                        "col": ci,
                        "text": text,
                    }
                )

    return {
        "immutable_sections": immutable_sections,
        "variable_fields": list(variable_fields.values()),
    }


def get_placeholder(field_id: str) -> str:
    """Return the original placeholder string for *field_id* (e.g. ``{{name}}`` or ``<name>``)."""
    return _PLACEHOLDER_TEMPLATES.get(field_id, "{{" + field_id + "}}")


# ---------------------------------------------------------------------------
# Template filling — supports both {{var}} and <var> placeholders
# ---------------------------------------------------------------------------


def _replace_in_text(text: str, field_id: str, replacement: str) -> str:
    """Replace both ``{{field_id}}`` and ``<field_id>`` in *text*."""
    text = text.replace("{{" + field_id + "}}", replacement)
    text = text.replace("<" + field_id + ">", replacement)
    return text


def fill_template(
    template_path: str,
    content_mapping: dict[str, str],
    output_path: str | None = None,
) -> str:
    """Replace ``{{field}}`` and ``<field>`` placeholders in a copy of *template_path*.

    Returns the path of the saved document.
    """
    doc = Document(template_path)

    # Replace in paragraphs
    for para in doc.paragraphs:
        original_text = para.text
        new_text = original_text
        for field_id, replacement in content_mapping.items():
            new_text = _replace_in_text(new_text, field_id, replacement)

        if new_text != original_text:
            # Apply change to runs — clear all runs and set first run text
            placeholder = get_placeholder(field_id)
            for run in para.runs:
                if placeholder in run.text or any(
                    ph in run.text for ph in [f"{{{{{field_id}}}}}", f"<{field_id}>"]
                ):
                    run.text = run.text.replace(
                        "{{" + field_id + "}}", replacement
                    ).replace("<" + field_id + ">", replacement)

    # Replace in tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                original_text = cell.text
                new_text = original_text
                for field_id, replacement in content_mapping.items():
                    new_text = _replace_in_text(new_text, field_id, replacement)

                if new_text != original_text:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            run.text = run.text.replace(
                                "{{" + field_id + "}}", replacement
                            ).replace("<" + field_id + ">", replacement)

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(config.output_dir, f"output_{ts}.docx")

    doc.save(output_path)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# DeepSeek LLM call (with retry & fallback) — ReAct-style observability
# ---------------------------------------------------------------------------


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    step_name: str = "llm_call",
) -> str:
    """Call DeepSeek (OpenAI-compatible) with retry + fallback + ReAct logging."""
    import openai

    models_to_try = [config.deepseek_model, config.deepseek_fallback_model]
    # deduplicate while preserving order
    models_to_try = list(dict.fromkeys(models_to_try))

    last_exc: Exception | None = None

    print(f"  Thought: Need to call LLM (model={models_to_try[0]})")
    print(f"  Action: _call_llm[{step_name}]")

    for attempt in range(config.max_retries):
        model = models_to_try[0] if attempt == 0 else models_to_try[-1]
        try:
            client = openai.OpenAI(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
                timeout=config.request_timeout,
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            print(
                f"  Observation: LLM responded successfully "
                f"({len(content)} chars, model={model})"
            )
            return content

        except Exception as exc:
            last_exc = exc
            print(
                f"  [WARN] LLM call failed (attempt {attempt + 1}, model={model}): "
                f"{exc}"
            )
            if attempt < config.max_retries - 1:
                delay = config.retry_base_delay * (2**attempt)
                print(f"  Observation: will retry in {delay:.0f}s (exponential backoff) …")
                time.sleep(delay)

    raise RuntimeError(
        f"LLM call failed after {config.max_retries} retries. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Code analysis
# ---------------------------------------------------------------------------

_ANALYZE_CODE_SYSTEM = (
    "You are an expert programming tutor. Given source code, "
    "explain: 1) what the code does, "
    "2) its key algorithms / logic, "
    "3) what output or result it produces. "
    "Be concise but thorough."
)


def analyze_code(code: str) -> str:
    """Analyze source code and return a structured explanation."""
    prompt = (
        "Analyze the following code and explain:\n"
        "- Purpose and functionality\n"
        "- Key algorithms and data structures\n"
        "- Expected output / results\n\n"
        f"```\n{code}\n```"
    )
    return _call_llm(
        system_prompt=_ANALYZE_CODE_SYSTEM,
        user_prompt=prompt,
        temperature=0.2,
        step_name="analyze_code",
    )


# ---------------------------------------------------------------------------
# Report content generation
# ---------------------------------------------------------------------------

_GENERATE_SYSTEM = (
    "You are an AI assistant that helps college students write lab reports. "
    "You will receive:\n"
    "- The lab report template structure (immutable sections)\n"
    "- The list of fields to fill (variable fields)\n"
    "- The user's requirements for the report\n"
    "- Code analysis (if applicable)\n"
    "- Similar past examples (few-shot)\n\n"
    "Generate professional, academically appropriate content for each variable "
    "field. The content must be detailed, accurate, and suitable for a "
    "college-level lab report.\n\n"
    "IMPORTANT:\n"
    "- DO NOT change, rephrase, or rewrite the immutable sections.\n"
    "- Only produce content for the listed variable fields.\n"
    "- Each field should contain complete paragraphs.\n"
    "- If the field expects code output, show realistic output.\n"
    "- Return ONLY a JSON object with field_id as key and content as value."
)


def generate_report(
    report_requirements: str,
    code_analysis: str,
    template_info: dict[str, Any],
    few_shot_examples: list[dict] | None = None,
) -> dict[str, str]:
    """Generate content for each variable field via DeepSeek."""
    # Build the user prompt
    sections_text = "\n".join(
        f"  [{s['type']}] {s['text'][:200]}"
        for s in template_info.get("immutable_sections", [])
    )
    fields_text = "\n".join(
        f"  - {f['field_id']}  (context: {f['context'][:150]})"
        for f in template_info.get("variable_fields", [])
    )

    few_shot_text = ""
    if few_shot_examples:
        parts = []
        for i, ex in enumerate(few_shot_examples, 1):
            parts.append(
                f"Example {i}:\n  Requirement: {ex['requirement']}\n"
                f"  Content: {ex['content']}"
            )
        few_shot_text = "--- Similar past reports ---\n" + "\n\n".join(parts)

    user_prompt = (
        f"--- Template immutable sections ---\n{sections_text}\n\n"
        f"--- Variable fields to fill ---\n{fields_text}\n\n"
        f"--- User requirements ---\n{report_requirements}\n\n"
        f"--- Code analysis ---\n{code_analysis}\n\n"
        f"{few_shot_text}\n\n"
        "Return a valid JSON object where keys are field_id and values are the "
        "generated content. Example:\n"
        '{"purpose": "The purpose of this experiment is ...", '
        '"result": "The output of the program is ..."}\n\n'
        "IMPORTANT: Return ONLY the JSON object, no other text."
    )

    raw = _call_llm(
        system_prompt=_GENERATE_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=8192,
        step_name="generate_report",
    )

    # Parse JSON from LLM output (handle ```json fences and stray text)
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find anything that looks like JSON
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                result = json.loads(raw[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                result = _fallback_parse(raw, template_info)
        else:
            result = _fallback_parse(raw, template_info)

    # Ensure all required fields are present
    for field in template_info.get("variable_fields", []):
        fid = field["field_id"]
        if fid not in result or not result[fid].strip():
            result[fid] = f"[Content for {fid} – see generated output below]"

    return result


def _fallback_parse(raw: str, template_info: dict[str, Any]) -> dict[str, str]:
    """If LLM output is not valid JSON, try to extract content heuristically."""
    result: dict[str, str] = {}
    for field in template_info.get("variable_fields", []):
        fid = field["field_id"]
        pattern = re.compile(
            rf"{re.escape(fid)}\s*[:：]\s*(.+?)(?=\n\s*\w+\s*[:：]|\Z)", re.DOTALL
        )
        m = pattern.search(raw)
        if m:
            result[fid] = m.group(1).strip()
        else:
            result[fid] = f"[Generated content for {fid}]"
    return result
