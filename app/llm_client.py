import logging

from openai import OpenAI

from .config import LLMConfig

llm_cfg = LLMConfig()
logger = logging.getLogger(__name__)

client = OpenAI(
    base_url=f"{llm_cfg.api_base}/v1",
    api_key=llm_cfg.api_key,
    timeout=llm_cfg.timeout_seconds,
)