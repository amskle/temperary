"""Configuration for LabReportAgent."""

import logging
import os
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)


@dataclass
class Config:
    # DeepSeek API
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_base_url: str = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
    )

    # Primary model: user wants deepseek-v4-pro by default
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

    # Fallback model: flash variant when pro rate-limits or times out
    deepseek_fallback_model: str = os.getenv(
        "DEEPSEEK_FALLBACK_MODEL", "deepseek-v4-flash"
    )

    # Retry & timeout (exponential backoff, up to 3 attempts)
    max_retries: int = 3
    retry_base_delay: float = 2.0  # seconds, doubled each attempt
    request_timeout: int = 120  # seconds

    # Vector memory
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    retrieval_top_k: int = 3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    memory_store_threshold: int = 4  # minimum rating to store memory

    # Generation
    max_generation_retries: int = 2
    prior_content_summary_threshold: int = 3000  # chars; summarise prior sections when exceeded

    # Code execution sandbox
    code_execution_timeout: int = 30  # seconds
    max_code_output_chars: int = 5000

    # Validation
    min_word_count: int = 30  # minimum words for valid academic content
    enable_llm_judge: bool = True  # use LLM-as-judge for borderline validation

    # Fields that need deep academic content (iterative single-field generation)
    complex_field_ids: set[str] = field(
        default_factory=lambda: {
            "purpose", "principle", "steps", "result", "analysis", "conclusion"
        }
    )

    # Output
    output_dir: str = os.getenv("OUTPUT_DIR", ".")


config = Config()

# Validate on import
if not config.deepseek_api_key:
    logging.warning(
        "DEEPSEEK_API_KEY is not set. "
        "Use 'export DEEPSEEK_API_KEY=sk-...'"
    )
