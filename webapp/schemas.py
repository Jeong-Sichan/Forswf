"""Public request and response shapes for the browser chat."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


Category = Literal["all", "culture", "etiquette", "food", "transit", "life"]
Language = Literal["auto", "en", "ko"]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1000)

    @field_validator("content")
    @classmethod
    def content_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=8)
    category: Category = "all"
    language: Language = "auto"

    @field_validator("message")
    @classmethod
    def message_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be blank")
        return value.strip()


class Source(BaseModel):
    title: str
    url: str
    license: str
    updated_at: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    category: Category
    language: Literal["en", "ko"]
    used_adapter: bool
    source_type: Literal["official", "heritage", "encyclopedia", "general", "none"]
