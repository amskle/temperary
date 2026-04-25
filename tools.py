"""Tools: template parsing, filling, code analysis, report generation."""

import json
import os
import re
import subprocess
import tempfile
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

    # Replace in paragraphs — iterate fields inside runs so each field gets replaced
    for para in doc.paragraphs:
        for field_id, replacement in content_mapping.items():
            for run in para.runs:
                if ("{{" + field_id + "}}" in run.text
                        or "<" + field_id + ">" in run.text):
                    run.text = _replace_in_text(run.text, field_id, replacement)

    # Replace in tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for field_id, replacement in content_mapping.items():
                    for para in cell.paragraphs:
                        for run in para.runs:
                            if ("{{" + field_id + "}}" in run.text
                                    or "<" + field_id + ">" in run.text):
                                run.text = _replace_in_text(
                                    run.text, field_id, replacement
                                )

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
# Dynamic code execution sandbox
# ---------------------------------------------------------------------------


def execute_code_sandbox(code: str, timeout: int | None = None) -> dict[str, str]:
    """Execute Python code in a sandboxed subprocess and capture actual output.

    Returns dict with keys: stdout, stderr, exit_code, error.
    The captured output is used to ground the LLM's result/analysis in real data.
    """
    if timeout is None:
        timeout = config.code_execution_timeout

    result: dict[str, str] = {
        "stdout": "",
        "stderr": "",
        "exit_code": "-1",
        "error": "",
    }

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        proc = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        max_chars = config.max_code_output_chars
        result["stdout"] = proc.stdout[:max_chars]
        result["stderr"] = proc.stderr[:max_chars]
        result["exit_code"] = str(proc.returncode)

    except subprocess.TimeoutExpired:
        result["error"] = f"Code execution timed out after {timeout}s"
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return result


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
    "- Code execution output (if applicable)\n"
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

_GENERATE_SINGLE_FIELD_SYSTEM = (
    "You are an AI assistant helping to write ONE specific section of a college "
    "lab report. Generate detailed, academically rigorous content for this "
    "section ONLY. The content must be substantive — include theoretical "
    "explanations, quantitative analysis, and domain-specific terminology "
    "appropriate for college-level coursework.\n\n"
    "CRITICAL RULES:\n"
    "- Write 2-5 substantial paragraphs with complete sentences.\n"
    "- Include specific details, equations, or data references where relevant.\n"
    "- Reference code execution output or experimental data when provided.\n"
    "- NEVER output placeholders, brackets, or generic fallback text.\n"
    "- NEVER write '[Content for...]' or any similar placeholder.\n"
    "- Return ONLY the raw content text — no JSON wrapper, no field ID label."
)

_GENERATE_SIMPLE_FIELDS_SYSTEM = (
    "You are an AI assistant filling in simple metadata fields for a lab report. "
    "Generate appropriate values for fields like student name, ID, date, etc.\n"
    "Return ONLY a JSON object with field_id as key and value as content."
)

# Fields that benefit from focused, single-field generation
_COMPLEX_FIELD_IDS = {"purpose", "principle", "steps", "result", "analysis", "conclusion"}


def _build_few_shot_text(few_shot_examples: list[dict] | None) -> str:
    """Format few-shot examples into a prompt-ready text block."""
    if not few_shot_examples:
        return ""
    parts = []
    for i, ex in enumerate(few_shot_examples, 1):
        parts.append(
            f"Example {i}:\n  Requirement: {ex['requirement']}\n"
            f"  Content: {ex['content']}"
        )
    return "--- Similar past reports ---\n" + "\n\n".join(parts)


def _build_immutable_context(template_info: dict[str, Any]) -> str:
    """Build the immutable sections context string from template info."""
    return "\n".join(
        f"  [{s['type']}] {s['text'][:3000]}"
        for s in template_info.get("immutable_sections", [])
    )


def _generate_simple_fields(
    simple_fields: list[dict],
    immutable_context: str,
    report_requirements: str,
    few_shot_text: str,
) -> dict[str, str]:
    """Generate simple metadata fields (name, student_id, date) in a single batch."""
    if not simple_fields:
        return {}

    fields_desc = "\n".join(
        f"  - {f['field_id']}  (context: {f['context'][:3000]})"
        for f in simple_fields
    )
    user_prompt = (
        f"--- Template structure ---\n{immutable_context}\n\n"
        f"--- Simple fields to fill ---\n{fields_desc}\n\n"
        f"--- User requirements ---\n{report_requirements}\n\n"
        f"{few_shot_text}\n\n"
        "Return a JSON object with field_id as key and content as value."
    )

    raw = _call_llm(
        system_prompt=_GENERATE_SIMPLE_FIELDS_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=1024,
        step_name="generate_simple_fields",
    )

    # Parse JSON
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                result = json.loads(raw[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                result = {}
        else:
            result = {}

    # Ensure all requested fields are present
    out: dict[str, str] = {}
    for f in simple_fields:
        fid = f["field_id"]
        out[fid] = result.get(fid, "").strip() if result.get(fid) else ""

    return out


def _generate_single_field(
    field: dict,
    immutable_context: str,
    report_requirements: str,
    code_analysis: str,
    code_execution_output: str,
    few_shot_text: str,
    prior_content: dict[str, str],
) -> str:
    """Generate one complex field with full context including prior sections."""
    fid = field["field_id"]
    context = field.get("context", "")[:3000]

    # Build prior content summary for context chaining
    prior_text = ""
    if prior_content:
        prior_lines = []
        for pk, pv in prior_content.items():
            preview = pv[:500].replace("\n", " ")
            prior_lines.append(f"  [{pk}]: {preview}")
        prior_text = (
            "--- Previously generated sections (read these to maintain "
            "consistency and avoid contradiction) ---\n"
            + "\n".join(prior_lines)
            + "\n\n"
        )

    code_section = ""
    if code_analysis:
        code_section += f"--- Code analysis ---\n{code_analysis}\n\n"
    if code_execution_output and code_execution_output.strip():
        code_section += (
            f"--- Actual code execution output (ground truth) ---\n"
            f"{code_execution_output}\n\n"
        )

    user_prompt = (
        f"--- Template structure (for context) ---\n{immutable_context}\n\n"
        f"--- Field to generate ---\n"
        f"Field ID: {fid}\n"
        f"Template context: {context}\n\n"
        f"{prior_text}"
        f"--- User requirements ---\n{report_requirements}\n\n"
        f"{code_section}"
        f"{few_shot_text}\n\n"
        f"Generate the complete content for the \"{fid}\" section. "
        f"Write 2-5 detailed paragraphs with substantive academic content. "
        f"Return ONLY the raw content — no labels, no JSON."
    )

    raw = _call_llm(
        system_prompt=_GENERATE_SINGLE_FIELD_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.4,
        max_tokens=4096,
        step_name=f"generate_field_{fid}",
    )

    return raw.strip()


def generate_report_iterative(
    report_requirements: str,
    code_analysis: str,
    code_execution_output: str,
    template_info: dict[str, Any],
    few_shot_examples: list[dict] | None = None,
) -> dict[str, str]:
    """Generate report content iteratively — one complex field at a time.

    Simple fields (name, student_id, date) are batched. Complex fields
    (purpose, principle, steps, result, analysis, conclusion) are generated
    individually with all prior content passed as context. This forces the
    LLM to focus its full attention and output tokens on one section at a time.
    """
    variable_fields = template_info.get("variable_fields", [])

    # Separate simple vs complex, preserving template order
    simple_fields = [
        f for f in variable_fields if f["field_id"] not in _COMPLEX_FIELD_IDS
    ]
    complex_fields = [
        f for f in variable_fields if f["field_id"] in _COMPLEX_FIELD_IDS
    ]

    immutable_context = _build_immutable_context(template_info)
    few_shot_text = _build_few_shot_text(few_shot_examples)

    result: dict[str, str] = {}

    # Phase 1: batch-generate simple metadata fields
    if simple_fields:
        simple_result = _generate_simple_fields(
            simple_fields, immutable_context, report_requirements, few_shot_text
        )
        result.update(simple_result)

    # Phase 2: generate each complex field individually with chained context
    for field in complex_fields:
        fid = field["field_id"]
        print(f"  Action: Generating field '{fid}' (iterative) …")
        generated = _generate_single_field(
            field=field,
            immutable_context=immutable_context,
            report_requirements=report_requirements,
            code_analysis=code_analysis,
            code_execution_output=code_execution_output,
            few_shot_text=few_shot_text,
            prior_content=result,
        )
        result[fid] = generated
        print(f"  Observation: '{fid}' generated ({len(generated)} chars)")

    # Fill any missing fields
    for field in variable_fields:
        fid = field["field_id"]
        if fid not in result or not result[fid].strip():
            result[fid] = f"[Content for {fid} — see generated output below]"

    return result


def generate_report(
    report_requirements: str,
    code_analysis: str,
    template_info: dict[str, Any],
    few_shot_examples: list[dict] | None = None,
    code_execution_output: str = "",
) -> dict[str, str]:
    """Generate content for each variable field via DeepSeek (single-pass fallback)."""
    # Build the user prompt — full context (no aggressive truncation)
    sections_text = "\n".join(
        f"  [{s['type']}] {s['text'][:3000]}"
        for s in template_info.get("immutable_sections", [])
    )
    fields_text = "\n".join(
        f"  - {f['field_id']}  (context: {f['context'][:3000]})"
        for f in template_info.get("variable_fields", [])
    )

    few_shot_text = _build_few_shot_text(few_shot_examples)

    code_output_section = ""
    if code_execution_output and code_execution_output.strip():
        code_output_section = (
            f"--- Actual code execution output (ground truth) ---\n"
            f"{code_execution_output}\n\n"
        )

    user_prompt = (
        f"--- Template immutable sections ---\n{sections_text}\n\n"
        f"--- Variable fields to fill ---\n{fields_text}\n\n"
        f"--- User requirements ---\n{report_requirements}\n\n"
        f"--- Code analysis ---\n{code_analysis}\n\n"
        f"{code_output_section}"
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
