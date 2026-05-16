"""Smoke-test reasoners — kept registered after .ai() verification so they're
available as a safety net if real reasoners start misbehaving.

Currently registered:
  - smoke_echo:   no-LLM control-plane plumbing check
  - smoke_text:   .ai() with structured Pydantic output (text input)
  - smoke_vision: .ai() with image URL (multimodal — proves Qwen3-VL works on OpenRouter)

Verified live on 2026-05-16 against openrouter/qwen/qwen3-vl-32b-instruct.
"""

from typing import Optional

from agentfield import AgentRouter
from pydantic import BaseModel, Field

smoke_router = AgentRouter(prefix="", tags=["smoke"])


@smoke_router.reasoner(tags=["smoke"])
async def smoke_echo(message: str) -> dict:
    return {"original": message, "length": len(message), "ok": True}


class TextSmokeResult(BaseModel):
    one_sentence_summary: str
    sentiment: str = Field(description="positive | negative | neutral")
    word_count_estimate: int = Field(ge=0)
    confident: bool


@smoke_router.reasoner(tags=["smoke"])
async def smoke_text(text: str, model: Optional[str] = None) -> dict:
    result = await smoke_router.ai(
        system="You are a concise text analyst. Return ONLY the requested fields.",
        user=f"Analyze this text:\n\n{text}",
        schema=TextSmokeResult,
        model=model,
    )
    return result.model_dump()


class VisionSmokeResult(BaseModel):
    description: str
    dominant_colors: list[str]
    object_count: int = Field(ge=0)
    confident: bool


@smoke_router.reasoner(tags=["smoke"])
async def smoke_vision(image_url: str, model: Optional[str] = None) -> dict:
    """Vision smoke. `image_url` may be either an http(s) URL or an inline
    `data:image/...;base64,...` string — both work as positional args to .ai().
    """
    result = await smoke_router.ai(
        "Look at this image and analyze it. Be specific about what you see.",
        image_url,
        system="You are a careful visual analyst. Return ONLY the requested fields.",
        schema=VisionSmokeResult,
        model=model,
    )
    return result.model_dump()
