#!/usr/bin/env python3
"""Streamlit web interface for LabReportAgent — AI-powered lab report generator."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from agent import run_agent
from memory import add_memory
from tools import parse_template

# ---------------------------------------------------------------------------
# Log collector — captures logger output for display in the UI
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(levelname)s | %(name)s | %(message)s"


class LogCollector(logging.Handler):
    """Accumulate log records into a list so they can be rendered post-run."""

    def __init__(self) -> None:
        super().__init__()
        self.logs: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.logs.append(self.format(record))


# ---------------------------------------------------------------------------
# Workspace helper — persist UploadedFile objects to a temp directory
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {
    ".py", ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".java", ".kt",
    ".txt", ".log", ".md", ".xml", ".launch", ".yaml", ".yml", ".json",
    ".cmake", ".cfg", ".ini", ".toml", ".sh", ".bash", ".zsh",
}


def save_workspace(
    template_file,
    material_files: list,
) -> tuple[str, str, str]:
    """Persist uploaded files into a temp workspace.

    Returns:
        ``(template_path, code_str, workspace_root)``
    """
    root = os.path.join(
        tempfile.gettempdir(), f"labreport_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(root, exist_ok=True)

    # ---- template ----
    template_path = os.path.join(root, template_file.name)
    with open(template_path, "wb") as f:
        f.write(template_file.getbuffer())

    # ---- material files → concatenated code string ----
    code_parts: list[str] = []
    for mf in material_files:
        ext = Path(mf.name).suffix.lower()
        if ext not in _TEXT_EXTENSIONS:
            continue
        try:
            content = mf.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            continue
        code_parts.append(f"// --- {mf.name} ---\n{content}")

    return template_path, "\n\n".join(code_parts), root


# ---------------------------------------------------------------------------
# UI entry point
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LabReportAgent", page_icon="📄", layout="wide")

# ==========================  Sidebar  ==========================

with st.sidebar:
    st.title("📄 LabReportAgent")
    st.caption("AI 实验报告生成助手 — LangGraph + MiMo")
    st.divider()

    language = st.selectbox(
        "🔧 编程语言",
        ["Python", "C/C++", "Java", "ROS", "MATLAB", "其他"],
    )

    requirement_text = st.text_area(
        "📋 实验要求描述",
        placeholder=(
            "描述你的实验内容、目标和要求...\n\n"
            "例如：\n- 实验目的与背景\n- 实验原理\n- 实验步骤\n- 需要分析的要点"
        ),
        height=180,
    )

    st.divider()

    template_file = st.file_uploader(
        "📄 上传实验报告模板",
        type=["docx"],
        accept_multiple_files=False,
        key="template",
    )

    material_files = st.file_uploader(
        "📎 上传素材文件（代码、日志、数据等）",
        type=None,
        accept_multiple_files=True,
        key="materials",
        help="支持 .py .c .cpp .java .txt .log .xml .yaml .json 等文本文件",
    )

    st.divider()

    generate_btn = st.button(
        "🚀 开始生成报告",
        type="primary",
        use_container_width=True,
        disabled=not (template_file and requirement_text),
    )

# ==========================  Main Panel  ==========================

st.title("📝 实验报告工作区")

if generate_btn:
    template_path, code, _workspace = save_workspace(
        template_file, material_files or []
    )

    full_requirement = (
        f"编程语言: {language}\n\n实验要求:\n{requirement_text}"
    )

    # Wire up log collector
    collector = LogCollector()
    collector.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger()
    root.addHandler(collector)

    try:
        with st.status("🔄 正在生成实验报告...", expanded=True):
            output_path, filled_content = run_agent(
                template_path=template_path,
                user_requirement=full_requirement,
                code=code,
            )

        # ---- Success + download ----
        st.success("✅ 实验报告生成成功！")

        with open(output_path, "rb") as f:
            doc_bytes = f.read()

        st.download_button(
            label=f"📥 下载实验报告 ({os.path.basename(output_path)})",
            data=doc_bytes,
            file_name=os.path.basename(output_path),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
        )

        # ---- Content preview ----
        if filled_content:
            st.divider()
            st.subheader("📖 生成的报告内容预览")
            for field_id, content in filled_content.items():
                with st.expander(f"📝 {field_id}  ({len(content)} 字符)"):
                    st.text(content)

        # ---- Agent trace ----
        if collector.logs:
            st.divider()
            with st.expander("🔍 Agent 思考过程 (ReAct Trace)", expanded=False):
                st.code("\n".join(collector.logs), language="text")

        # ---- User rating ----
        st.divider()
        st.subheader("⭐ 给这份报告评分")
        st.caption("评分 ≥ 4 分会存入记忆库，下次生成类似报告时作为参考")
        rating = st.slider("评分", min_value=1, max_value=5, value=3, key="rating_slider")
        rating_labels = {1: "很差", 2: "较差", 3: "一般", 4: "不错", 5: "优秀"}
        st.write(f"当前评分: **{rating}** — {rating_labels[rating]}")

        if st.button("📩 提交评分", key="submit_rating"):
            try:
                # Extract template headers for better embedding
                template_headers = ""
                try:
                    info = parse_template(template_path)
                    template_headers = " | ".join(
                        f["field_id"] for f in info.get("variable_fields", [])
                    )
                except Exception:
                    pass

                add_memory(
                    requirement=full_requirement,
                    generated_content=filled_content,
                    rating=rating,
                    template_headers=template_headers,
                )
                if rating >= 4:
                    st.success(f"✅ 感谢评分！评分 {rating} 分已存入记忆库，将用于优化未来报告生成。")
                else:
                    st.info(f"ℹ️ 感谢评分！评分 {rating} 分（< 4 分不存入记忆库）。")
            except Exception as e:
                st.warning(f"评分保存失败: {e}")

    except Exception as exc:
        st.error(f"❌ 生成失败: {exc}")
        if collector.logs:
            with st.expander("🔍 查看详细日志"):
                st.code("\n".join(collector.logs), language="text")

    finally:
        root.removeHandler(collector)

else:
    # Landing state — show next steps + any uploaded file info
    st.info(
        "👈 请在左侧边栏上传模板并填写实验要求，"
        "然后点击「开始生成报告」按钮。"
    )

    if template_file:
        st.caption(
            f"已上传模板: **{template_file.name}** "
            f"({template_file.size:,} bytes)"
        )

    if material_files:
        st.caption(f"已上传 {len(material_files)} 个素材文件")
        for mf in material_files:
            st.caption(f"  • {mf.name} ({mf.size:,} bytes)")
