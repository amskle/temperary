"""Tools: template parsing, filling, code analysis, report generation."""

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Any

from docx import Document

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template parsing — detects {{var}} style Jinja2 placeholders
# ---------------------------------------------------------------------------

_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def _extract_placeholders(text: str) -> list[dict]:
    """Extract placeholder info from text."""
    results: list[dict] = []
    for match in _VARIABLE_PATTERN.finditer(text):
        field_id = match.group(1)
        results.append({"field_id": field_id, "raw_placeholder": match.group(0)})
    return results


def parse_template(template_path: str) -> dict[str, Any]:
    """Parse a .docx template, returning immutable sections and variable fields.

    Returns
        {
          "immutable_sections": [{"type": str, "index": int, "text": str}, ...],
          "variable_fields": [{"field_id": str, "placeholder": str,
                               "context": str}, ...]
        }

    Supports ``{{field}}`` style Jinja2 placeholders.
    """
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



# ---------------------------------------------------------------------------
# Template filling — DocxTemplate renders Jinja2 {{field}} placeholders at XML level
# ---------------------------------------------------------------------------


def fill_template(
    template_path: str,
    content_mapping: dict[str, str],
    output_path: str | None = None,
) -> str:
    """Render ``{{field}}`` placeholders with DocxTemplate and save.

    DocxTemplate works at the XML level, so it naturally handles
    placeholders split across multiple runs by Word.
    """
    from docxtpl import DocxTemplate

    doc = DocxTemplate(template_path)
    doc.render(content_mapping)

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

    logger.info("Thought: Need to call LLM (model=%s)", models_to_try[0])
    logger.info("Action: _call_llm[%s]", step_name)

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
            logger.info(
                "Observation: LLM responded successfully (%d chars, model=%s)",
                len(content), model,
            )
            return content

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "LLM call failed (attempt %d, model=%s): %s",
                attempt + 1, model, exc,
            )
            if attempt < config.max_retries - 1:
                delay = config.retry_base_delay * (2**attempt)
                logger.info(
                    "Observation: will retry in %.0fs (exponential backoff) …",
                    delay,
                )
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


def _extract_json_from_llm_response(raw: str) -> dict:
    """Extract and parse a JSON object from LLM output.

    Handles markdown code fences (`` ```json ... ``` ``) and stray text
    surrounding the JSON object. Returns an empty dict on failure.
    """
    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(raw[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass
        return {}


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

    result = _extract_json_from_llm_response(raw)

    # Ensure all requested fields are present
    out: dict[str, str] = {}
    for f in simple_fields:
        fid = f["field_id"]
        out[fid] = result.get(fid, "").strip() if result.get(fid) else ""

    return out


_SUMMARIZE_PRIOR_SYSTEM = (
    "You are a precise academic summarizer. Condense the following generated "
    "lab report sections into a concise, structured summary. Preserve key "
    "details: experimental values, equations, named methods, conclusions, "
    "and cross-references. The summary will be used as context for generating "
    "the NEXT section, so include anything needed for logical continuity."
)


def _summarize_prior_content(prior_text: str) -> str:
    """Compress long prior-content text into a summary via an LLM call."""
    raw = _call_llm(
        system_prompt=_SUMMARIZE_PRIOR_SYSTEM,
        user_prompt=(
            "Summarize the following lab report sections. "
            "Keep all quantitative data, named methods, equations, and key "
            "conclusions. Write a single flowing paragraph (≤300 words).\n\n"
            f"{prior_text}"
        ),
        temperature=0.2,
        max_tokens=1024,
        step_name="summarize_prior",
    )
    return raw.strip()


def _build_prior_context(prior_content: dict[str, str]) -> str:
    """Build prior-section context, summarising if the raw text is too long."""
    if not prior_content:
        return ""

    prior_lines = []
    for pk, pv in prior_content.items():
        preview = pv[:500].replace("\n", " ")
        prior_lines.append(f"  [{pk}]: {preview}")

    raw_text = (
        "--- Previously generated sections (read these to maintain "
        "consistency and avoid contradiction) ---\n"
        + "\n".join(prior_lines)
    )

    if len(raw_text) > config.prior_content_summary_threshold:
        logger.info(
            "  Prior content length %d exceeds threshold %d — summarising.",
            len(raw_text), config.prior_content_summary_threshold,
        )
        summary = _summarize_prior_content(raw_text)
        return (
            "--- Summary of previously generated sections (read for "
            "consistency and logical continuity) ---\n"
            f"{summary}\n\n"
        )

    return raw_text + "\n\n"


def _generate_single_field(
    field: dict,
    immutable_context: str,
    report_requirements: str,
    code_analysis: str,
    code_execution_output: str,
    few_shot_text: str,
    prior_content: dict[str, str],
    retry_feedback: str = "",
) -> str:
    """Generate one complex field with full context including prior sections."""
    fid = field["field_id"]
    context = field.get("context", "")[:3000]

    prior_text = _build_prior_context(prior_content)

    code_section = ""
    if code_analysis:
        code_section += f"--- Code analysis ---\n{code_analysis}\n\n"
    if code_execution_output and code_execution_output.strip():
        code_section += (
            f"--- Actual code execution output (ground truth) ---\n"
            f"{code_execution_output}\n\n"
        )

    retry_section = ""
    if retry_feedback:
        retry_section = (
            "--- IMPORTANT: Previous generation attempt failed ---\n"
            f"Issues to fix: {retry_feedback}\n"
            "Please ensure the content above is substantive, complete, and "
            "free of placeholder or generic fallback text.\n\n"
        )

    user_prompt = (
        f"--- Template structure (for context) ---\n{immutable_context}\n\n"
        f"--- Field to generate ---\n"
        f"Field ID: {fid}\n"
        f"Template context: {context}\n\n"
        f"{prior_text}"
        f"--- User requirements ---\n{report_requirements}\n\n"
        f"{code_section}"
        f"{retry_section}"
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
    retry_feedback: str = "",
) -> dict[str, str]:
    """Generate report content iteratively — one complex field at a time.

    Simple fields (name, student_id, date) are batched. Complex fields
    (purpose, principle, steps, result, analysis, conclusion) are generated
    individually with all prior content passed as context. This forces the
    LLM to focus its full attention and output tokens on one section at a time.

    If *retry_feedback* is non-empty, it is injected into each complex-field
    prompt so the LLM knows what failed the previous validation pass.
    """
    variable_fields = template_info.get("variable_fields", [])

    # Separate simple vs complex, preserving template order
    simple_fields = [
        f for f in variable_fields if f["field_id"] not in config.complex_field_ids
    ]
    complex_fields = [
        f for f in variable_fields if f["field_id"] in config.complex_field_ids
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
        logger.info("Action: Generating field '%s' (iterative) …", fid)
        generated = _generate_single_field(
            field=field,
            immutable_context=immutable_context,
            report_requirements=report_requirements,
            code_analysis=code_analysis,
            code_execution_output=code_execution_output,
            few_shot_text=few_shot_text,
            prior_content=result,
            retry_feedback=retry_feedback,
        )
        result[fid] = generated
        logger.info("Observation: '%s' generated (%d chars)", fid, len(generated))

    # Fill any missing fields
    for field in variable_fields:
        fid = field["field_id"]
        if fid not in result or not result[fid].strip():
            result[fid] = f"[Content for {fid} — see generated output below]"

    return result
