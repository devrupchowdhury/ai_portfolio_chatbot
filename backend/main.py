"""Ask My AI — backend.

FastAPI service that parses a resume at startup and answers recruiter-style
questions using the parsed resume data as its factual context.
"""

import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pypdf import PdfReader

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("ask-my-ai")

# Configuration ----------------------------------------------------------------

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
RESUME_PATH = Path(os.getenv("RESUME_PATH", "my_resume.pdf"))
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "2"))
MAX_RESUME_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_RESUME_PAGES = 20

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "https://devrupchowdhury.vercel.app"
    ).split(",")
    if origin.strip()
]

# Allows the production Vercel domain and project preview deployments.
# Set ALLOWED_ORIGIN_REGEX to an empty string to disable regex matching.
ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALLOWED_ORIGIN_REGEX",
    r"^https://devrupchowdhury-[a-zA-Z0-9]+-devrups-projects\.vercel\.app$",
).strip() or None

# Do not silently retain the old wildcard setting from the previous version.
# CORS is not authentication, but a broad wildcard makes browser access too open.
if "*" in ALLOWED_ORIGINS:
    logger.warning("Ignoring '*' in ALLOWED_ORIGINS; configure explicit trusted origins.")
    ALLOWED_ORIGINS = [origin for origin in ALLOWED_ORIGINS if origin != "*"]

_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    """Create the Groq client lazily, so the app can still expose health status
    when configuration is incomplete. The app will not become ready until the
    resume has been parsed successfully.
    """
    global _groq_client

    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")

        _groq_client = Groq(
            api_key=api_key,
            timeout=GROQ_TIMEOUT_SECONDS,
            max_retries=GROQ_MAX_RETRIES,
        )

    return _groq_client


# Pydantic schemas --------------------------------------------------------------

class Experience(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    model_config = ConfigDict(extra="ignore")
