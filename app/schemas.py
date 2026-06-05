from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptTurn(BaseModel):
    speaker: str
    time: Optional[str] = None
    text: str


class AnalyzeCallRequest(BaseModel):
    scenario: str = "billing"
    transcript: List[TranscriptTurn]


class QualityScores(BaseModel):
    empathy: int = Field(ge=0, le=100)
    clarity: int = Field(ge=0, le=100)
    compliance: int = Field(ge=0, le=100)
    risk_control: int = Field(ge=0, le=100)


class Topic(BaseModel):
    name: str
    explanation: str


class SuggestedAction(BaseModel):
    title: str
    description: str
    draft: str = ""


class AnalyzeCallResponse(BaseModel):
    summary: str
    sentiment_score: int = Field(ge=0, le=100)
    sentiment_label: str
    sentiment_delta: str
    resolution_confidence: int = Field(ge=0, le=100)
    priority: str
    priority_reason: str
    churn_signal: str
    churn_reason: str
    quality_scores: QualityScores
    topics: List[Topic]
    suggested_actions: List[SuggestedAction]
    ticket_draft: str
    email_draft: str
    crm_note: str