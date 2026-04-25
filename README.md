# LabReportAgent

An AI-powered lab report generator for college students. Upload a `.docx` template, describe your requirements (and optionally provide source code), and get a professionally filled lab report — while the template's fixed structure stays **untouched**.

## Architecture

```
┌─────────────┐    ┌──────────────────────┐    ┌───────────────┐
│   main.py   │───▶│    agent.py          │───▶│  tools.py     │
│  (CLI arg   │    │  (LangGraph state    │    │  parse / fill │
│   parser)   │    │   machine)           │    │  analyze /    │
└─────────────┘    │                      │    │  generate     │
                   │  Nodes:              │    └───────┬───────┘
                   │  1. parse_template   │            │
                   │  2. analyze_code     │    ┌───────▼───────┐
                   │  3. generate_content │    │  memory.py    │
                   │  4. fill_and_save    │    │  Chroma +     │
                   └──────────────────────┘    │  sentence-    │
                                               │  transformers │
                                               └───────────────┘
```

### Flow

1. **Parse template** — reads the `.docx`, identifies `{{variable}}` fields vs. immutable sections.
2. **Retrieve memory** — vector-search past similar requests for few-shot examples.
3. **Analyze code** (optional) — if source code is provided, LLM explains its purpose & output.
4. **Generate content** — calls DeepSeek to produce content for each variable field.
5. **Validate & retry** — if content is empty or too short, retries (up to 3×).
6. **Fill & save** — replaces placeholders in a copy of the template, saves the result.

## Requirements

- Python 3.10+
- A [DeepSeek](https://platform.deepseek.com/) API key

## Setup

```bash
# 1. Clone / enter the project directory
cd LabReportAgent

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your DeepSeek API key
export DEEPSEEK_API_KEY="sk-..."
# Optional: change the base URL or model
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export DEEPSEEK_MODEL="deepseek-chat"
```

## Usage

```bash
# Basic: template + text requirement
python main.py --template templates/demo.docx --requirement "写一份关于二分查找算法的实验报告"

# With source code (the agent analyses the code and includes results)
python main.py --template templates/demo.docx \
    --requirement "快速排序算法实验报告" \
    --code quicksort.py

# Requirement can be a file path instead of inline text
python main.py --template templates/demo.docx \
    --requirement requirements.txt \
    --code solution.py
```

After generation, the agent asks for a rating (1–5). Ratings ≥4 are stored in the vector database and used as few-shot examples in future runs.

## Template format

Create a `.docx` file with `{{variable_name}}` placeholders where content should be filled:

```
实验报告
姓名：{{name}}    学号：{{student_id}}

一、实验目的
{{purpose}}

二、实验原理
{{principle}}

三、实验结果
{{result}}
```

The agent fills only `{{…}}` placeholders. Everything else is treated as an immutable structural element.

## Project structure

```
LabReportAgent/
├── main.py              # CLI entry point
├── agent.py             # LangGraph state machine
├── tools.py             # Template parsing, filling, LLM calls
├── memory.py            # Chroma vector DB for few-shot memory
├── config.py            # Environment & model configuration
├── requirements.txt
├── templates/           # Place your .docx templates here
├── tests/
│   └── test_template_parser.py
└── README.md
```

## Running tests

```bash
pytest tests/
```

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | API key (required) |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | API base URL |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Primary model |
| `DEEPSEEK_FALLBACK_MODEL` | `deepseek-chat` | Fallback model |
| `CHROMA_DB_PATH` | `./chroma_db` | Vector DB directory |
| `OUTPUT_DIR` | `.` | Output directory for generated reports |
