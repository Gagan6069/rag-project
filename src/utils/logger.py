import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

for noisy_logger in (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger("RAG")
