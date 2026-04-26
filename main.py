#!/usr/bin/env python3
"""CLI entry point for LabReportAgent."""

import argparse
import logging
import os
import sys

from agent import run_agent
from memory import add_memory
from tools import parse_template

logger = logging.getLogger(__name__)


def load_text(path_or_text: str) -> str:
    """If *path_or_text* is an existing file, read it; otherwise return as-is."""
    if os.path.isfile(path_or_text):
        with open(path_or_text, encoding="utf-8") as f:
            return f.read()
    return path_or_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LabReportAgent – AI-powered lab report generator."
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to the .docx template file.",
    )
    parser.add_argument(
        "--requirement",
        "-r",
        required=True,
        help="Report requirement text, or path to a file containing it.",
    )
    parser.add_argument(
        "--code",
        "-c",
        default="",
        help="Optional source code file path (or inline code).",
    )
    args = parser.parse_args()

    # Validate template
    if not os.path.isfile(args.template):
        logger.error("Template not found: %s", args.template)
        sys.exit(1)

    requirement = load_text(args.requirement)
    code = load_text(args.code) if args.code else ""

    logger.info("=" * 60)
    logger.info("  LabReportAgent – AI Lab Report Generator")
    logger.info("  Powered by LangGraph + DeepSeek")
    logger.info("=" * 60)
    logger.info("  Template   : %s", args.template)
    logger.info("  Requirement: %s", requirement[:120] + ('…' if len(requirement) > 120 else ''))
    logger.info("  Code       : %s", 'Yes (' + str(len(code)) + ' chars)' if code else 'No')
    logger.info("=" * 60)

    try:
        output_path, filled_content = run_agent(
            template_path=args.template,
            user_requirement=requirement,
            code=code,
        )

        logger.info("=" * 60)
        logger.info("  ✅ Report generated: %s", output_path)
        logger.info("=" * 60)

        # Feedback loop — store generated content with template context
        _collect_feedback(
            requirement, filled_content, template_path=args.template
        )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error("%s", e)
        sys.exit(1)


def _collect_feedback(
    requirement: str, filled_content: dict[str, str], template_path: str = ""
) -> None:
    """Ask user for a rating and store memory if appropriate."""
    logger.info("-" * 40)
    logger.info("  Feedback (rate the generated report)")
    logger.info("-" * 40)
    logger.info("  Rating: 1 (poor) – 5 (excellent), or 0 to skip.")

    try:
        rating_str = input("  Your rating: ").strip()
        rating = int(rating_str)
    except (ValueError, EOFError):
        return

    if rating <= 0 or rating > 5:
        logger.info("  Skipped.")
        return

    # Extract template headers for better embedding accuracy
    template_headers = ""
    if template_path:
        try:
            info = parse_template(template_path)
            template_headers = " | ".join(
                f["field_id"] for f in info.get("variable_fields", [])
            )
        except Exception:
            pass

    try:
        add_memory(
            requirement=requirement,
            generated_content=filled_content,
            rating=rating,
            template_headers=template_headers,
        )
        if rating >= 4:
            logger.info("  ✅ Stored in memory (rating=%d).", rating)
        else:
            logger.info("  ℹ️  Not stored (rating=%d < 4).", rating)
    except Exception as e:
        logger.warning("  Failed to store memory: %s", e)


if __name__ == "__main__":
    main()
