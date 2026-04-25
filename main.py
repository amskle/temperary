#!/usr/bin/env python3
"""CLI entry point for LabReportAgent."""

import argparse
import os
import sys
from datetime import datetime

from agent import run_agent
from memory import add_memory
from tools import parse_template


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
        print(f"[ERROR] Template not found: {args.template}")
        sys.exit(1)

    requirement = load_text(args.requirement)
    code = load_text(args.code) if args.code else ""

    print("=" * 60)
    print("  LabReportAgent – AI Lab Report Generator")
    print("  Powered by LangGraph + DeepSeek")
    print("=" * 60)
    print(f"  Template   : {args.template}")
    print(f"  Requirement: {requirement[:120]}{'…' if len(requirement) > 120 else ''}")
    print(f"  Code       : {'Yes (' + str(len(code)) + ' chars)' if code else 'No'}")
    print("=" * 60)

    try:
        output_path = run_agent(
            template_path=args.template,
            user_requirement=requirement,
            code=code,
        )

        print("\n" + "=" * 60)
        print(f"  ✅ Report generated: {output_path}")
        print("=" * 60)

        # Feedback loop — store generated content with template context
        _collect_feedback(requirement, output_path, template_path=args.template)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


def _collect_feedback(
    requirement: str, output_path: str, template_path: str = ""
) -> None:
    """Ask user for a rating and store memory if appropriate."""
    print("\n" + "-" * 40)
    print("  Feedback (rate the generated report)")
    print("-" * 40)
    print("  Rating: 1 (poor) – 5 (excellent), or 0 to skip.")

    try:
        rating_str = input("  Your rating: ").strip()
        rating = int(rating_str)
    except (ValueError, EOFError):
        return

    if rating <= 0 or rating > 5:
        print("  Skipped.")
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
            generated_content={"report_path": output_path},
            rating=rating,
            template_headers=template_headers,
        )
        if rating >= 4:
            print(f"  ✅ Stored in memory (rating={rating}).")
        else:
            print(f"  ℹ️  Not stored (rating={rating} < 4).")
    except Exception as e:
        print(f"  [WARN] Failed to store memory: {e}")


if __name__ == "__main__":
    main()
