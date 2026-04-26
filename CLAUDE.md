# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指导。

## 常用命令

```bash
# 安装依赖（如果代理有问题，先 unset 代理变量）
pip install -r requirements.txt

# 运行全部测试
pytest tests/

# 运行单个测试
pytest tests/test_template_parser.py -k test_identifies_variable_fields

# 运行 agent
python main.py --template templates/demo.docx --requirement "实验报告要求" --code path/to/code.py

# requirement 参数也可以是文件路径
python main.py --template templates/demo.docx -r requirements.txt
```

## 架构

Agent 是一个 **LangGraph ReAct 风格的状态机**，包含 4 个节点和一个验证关卡：

```
main.py (CLI) → agent.py (图) → tools.py (解析/填充/生成/执行)
                                  memory.py (ChromaDB 向量存储)
```

**图的流转：** `parse_and_retrieve` → `analyze_code_if_needed` → `generate_content` ⟲ `validate_content` → `fill_and_save`

### 关键设计决策

- **迭代生成**（`tools.py:generate_report_iterative`）：简单字段（姓名、学号、日期）批量生成。复杂字段（目的、原理、步骤、结果、分析、结论）逐个生成，每个字段都会收到之前已生成的所有章节作为链式上下文。这迫使 LLM 将全部输出 token 集中在一个章节上，并保持一致性。
- **双占位符风格**：同时支持 `{{var}}` 和 `<var>` 两种语法。每种字段的原始风格会被记住，并在输出时保持一致。
- **带验证的重试**：生成后，内容会经过占位符残留检测、最小长度检查和可选的 LLM 质量评分。最多重试 `max_generation_retries`（2 次）。
- **LLM 容错**（`tools.py:_call_llm`）：指数退避重试，失败时自动切换到备用模型。配置见 `config.py`。
- **代码执行沙箱**（`tools.py:execute_code_sandbox`）：用户提供的 Python 代码在子进程中运行，30 秒超时。真实的 stdout/stderr 被捕获并作为"真值"注入到生成 prompt 中。
- **少样本记忆**（`memory.py`）：用户评分 ≥4 时存入 ChromaDB。嵌入向量将模板字段标题与需求文本结合，使得检索能同时匹配主题内容和报告结构。检索到的示例会输入到生成 prompt 中。

### 配置（`config.py`）

所有设置通过环境变量配置，有合理的默认值。关键变量：
- `DEEPSEEK_API_KEY`（必填）
- `DEEPSEEK_MODEL`（默认：`deepseek-v4-pro`），`DEEPSEEK_FALLBACK_MODEL`（默认：`deepseek-v4-flash`）
- `CHROMA_DB_PATH`、`OUTPUT_DIR`
