import json
import logging
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .config import LLMConfig
from .llm_client import client
from .schemas import AnalyzeCallRequest

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "index.html"

llm_cfg = LLMConfig()

app = FastAPI(
    title="Customer Service Copilot Backend",
    version="1.0.0",
)


@app.get("/")
def serve_dashboard() -> FileResponse:
    logger.info("Looking for index.html at: %s", INDEX_PATH)

    if not INDEX_PATH.exists():
        existing_files = [str(p) for p in APP_DIR.glob("*")]

        raise HTTPException(
            status_code=404,
            detail={
                "message": "index.html not found",
                "expected_path": str(INDEX_PATH),
                "app_dir": str(APP_DIR),
                "files_in_app_dir": existing_files,
            },
        )

    return FileResponse(INDEX_PATH)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": "customer-service-copilot",
        "llm_api_base": llm_cfg.api_base,
        "llm_model": llm_cfg.model,
    }


@app.get("/api/llm-health")
def llm_health() -> dict[str, Any]:
    try:
        models = client.models.list()

        return {
            "status": "ok",
            "llm_api_base": llm_cfg.api_base,
            "models": [model.id for model in models.data],
        }
    except Exception as exc:
        logger.exception("LLM health check failed")

        raise HTTPException(
            status_code=502,
            detail={
                "message": "LLM server is not reachable from backend",
                "llm_api_base": llm_cfg.api_base,
                "error": str(exc),
            },
        )


@app.post("/api/analyze-call")
def analyze_call(payload: AnalyzeCallRequest) -> dict[str, Any]:
    if not payload.transcript:
        raise HTTPException(
            status_code=400,
            detail="Transcript cannot be empty",
        )

    transcript_text = "\n".join(
        f"{turn.speaker} [{turn.time or '-'}]: {turn.text}"
        for turn in payload.transcript
    )

    prompt = build_prompt(
        scenario=payload.scenario,
        transcript_text=transcript_text,
    )

    try:
        logger.info(
            "Sending call transcript to LLM. api_base=%s model=%s",
            llm_cfg.api_base,
            llm_cfg.model,
        )

        response = client.chat.completions.create(
            model=llm_cfg.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Customer Service Copilot. "
                        "You analyze completed customer service calls. "
                        "You do not talk to customers directly. "
                        "You prepare drafts and suggestions for human representatives and supervisors."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_tokens=1800,
        )

        content = response.choices[0].message.content or ""
        parsed = parse_json_response(content)

        return parsed

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("LLM call analysis failed")

        raise HTTPException(
            status_code=500,
            detail={
                "message": "LLM call analysis failed",
                "llm_api_base": llm_cfg.api_base,
                "model": llm_cfg.model,
                "error": str(exc),
            },
        )


def build_prompt(scenario: str, transcript_text: str) -> str:
    return f"""
Analyze the following completed customer service call.

Important positioning:
- The human customer service representative talks to the customer.
- You are only a post-call copilot.
- You do NOT talk to the customer directly.
- You prepare drafts and suggestions for a human representative or supervisor.
- Do not claim that actions are already executed.
- Keep outputs short, professional and enterprise-friendly.

Scenario:
{scenario}

Transcript:
{transcript_text}

Return ONLY valid JSON. Do not wrap it in markdown.

Use exactly this JSON structure:

{{
  "summary": "professional post-call summary",
  "sentiment_score": 0,
  "sentiment_label": "Negative | Neutral | Mostly Positive | At Risk | Positive",
  "sentiment_delta": "short sentiment change explanation",
  "resolution_confidence": 0,
  "priority": "P1 | P2 | P3",
  "priority_reason": "short reason",
  "churn_signal": "Low | Medium | High",
  "churn_reason": "short reason",
  "quality_scores": {{
    "empathy": 0,
    "clarity": 0,
    "compliance": 0,
    "risk_control": 0
  }},
  "topics": [
    {{
      "name": "topic name",
      "explanation": "why this topic was detected"
    }}
  ],
  "suggested_actions": [
    {{
      "title": "suggested action title",
      "description": "what the representative or supervisor should review",
      "draft": "short draft text if relevant"
    }}
  ],
  "ticket_draft": "short ticket draft",
  "email_draft": "short customer follow-up email draft",
  "crm_note": "short CRM note draft"
}}

Rules:
- sentiment_score must be 0-100.
- resolution_confidence must be 0-100.
- quality score values must be 0-100.
- suggested_actions should contain 3 to 5 items.
- topics should contain 3 to 6 items.
- Use English.
"""


def parse_json_response(content: str) -> dict[str, Any]:
    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)

        if not match:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "LLM did not return valid JSON",
                    "raw_response": content,
                },
            )

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "LLM returned JSON-like text but parsing failed",
                    "error": str(exc),
                    "raw_response": content,
                },
            )