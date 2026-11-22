"""
Ask My AI — backend
FastAPI service that parses a resume once at startup and answers
recruiter/HR-style questions about the candidate, grounded strictly
in the parsed resume data (no hallucination beyond what's on file).
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ask-my-ai")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
RESUME_PATH = Path(os.getenv("RESUME_PATH", "my_resume.pdf"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
# Matches your stable domain (devrupchowdhury.vercel.app) AND any of
# Vercel's auto-generated preview deployment URLs for the same project
# (e.g. devrupchowdhury-7k1wx9sld-devrups-projects.vercel.app), so you
# don't have to update ALLOWED_ORIGINS every time Vercel creates a new
# preview URL. Override via the ALLOWED_ORIGIN_REGEX env var if your
# Vercel project name differs.
ALLOWED_ORIGIN_REGEX = os.getenv(
    "ALLOWED_ORIGIN_REGEX",
    r"^https://devrupchowdhury(-[a-zA-Z0-9]+)*\.vercel\.app$",
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Create a .env file (see .env.example)."
    )

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Ask My AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------- Schemas ----------

class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


RESUME_SCHEMA = Resume.model_json_schema()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question cannot be empty or whitespace")
        return v


class ChatResponse(BaseModel):
    answer: str


# ---------- Resume parsing ----------

def read_pdf(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Resume file not found: {file_path}")

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        raise ValueError(f"No extractable text found in {file_path}")

    return text


def parse_resume(resume_text: str) -> Resume:
    system_prompt = f"""
You are an expert resume parser.

Extract information based on meaning, not literal section headings.
Different resumes label the same content differently — Experience,
Professional Experience, Work History, Employment, and Internships
all represent relevant work experience. Skills may appear in a
dedicated skills section, or embedded within experience, projects,
certifications, or summaries anywhere in the resume.

Return ONLY valid JSON matching this schema, with no other text,
no explanations, and no Markdown formatting or code fences:

{json.dumps(RESUME_SCHEMA)}

Rules:
1. Do not invent, infer, or guess information not present in the resume.
2. If a field's value is not available, return null.
3. If a list-type field has no items, return an empty list, never null.
4. Treat internships as experience entries; include them in the experiences list.
5. Collect skills mentioned anywhere in the resume, not just from a skills section, and deduplicate them.
6. Preserve dates, titles, and organization names exactly as written in the resume; do not reformat, rephrase, or normalize them unless the schema explicitly requires a specific format.
7. If the same information appears in multiple places (e.g. a skill listed both in a skills section and a project description), include it only once.
8. Output must be a single valid JSON object parseable by a standard JSON parser — no trailing commas, no comments, no surrounding text.
"""


    user_prompt = f"Parse the following resume:\n\n{resume_text}"

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw_output = response.choices[0].message.content
        data = json.loads(raw_output)
        return Resume(**data)
    except json.JSONDecodeError as e:
        logger.error("Resume parser returned invalid JSON: %s", e)
        raise ValueError("Resume parsing failed: model returned malformed JSON") from e
    except Exception as e:
        logger.error("Resume parsing failed: %s", e)
        raise


def ask_candidate(question: str, resume: Resume) -> str:
    system_prompt = f"""You are an AI assistant representing a job candidate.

Below is everything you know about the candidate:

{resume.model_dump_json(indent=2)}

Rules:
1. Answer only using the information provided above.
2. Never hallucinate or invent details that are not present.
3. If the information is unavailable, say exactly:
   "I don't have enough information to answer that."
4. Be professional, concise, and natural.
5. Answer as if you are an HR interviewer evaluating this candidate.
6. Answer the question directly. Do not add unnecessary introductions or conclusions.
7. Do not repeat the user's question.
8. Do not use Markdown formatting.
9. Do not use *, **, #, bullet points, or numbered lists.
10. Use plain text with clean paragraphs.
11. If multiple technologies or skills need to be mentioned, separate them naturally using commas and short paragraphs.
12. Do not make assumptions about the candidate's experience or abilities.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Groq chat completion failed: %s", e)
        raise


# ---------- App state ----------

resume_cache: Resume | None = None


@app.on_event("startup")
def load_resume() -> None:
    global resume_cache
    try:
        resume_text = read_pdf(RESUME_PATH)
        resume_cache = parse_resume(resume_text)
        logger.info("Resume loaded and parsed successfully: %s", RESUME_PATH)
    except Exception as e:
        # Don't crash the whole app on startup — surface the error per-request
        # instead, so you can fix the resume file without a full redeploy.
        logger.error("Failed to load resume at startup: %s", e)
        resume_cache = None


# ---------- Routes ----------

@app.get("/")
def home():
    return {
        "message": "Ask My AI is running.",
        "resume_loaded": resume_cache is not None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "resume_loaded": resume_cache is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if resume_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Resume data is not loaded. Check server logs and RESUME_PATH.",
        )

    try:
        answer = ask_candidate(request.question, resume_cache)
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The AI service failed to generate a response. Try again shortly.",
        )

    return ChatResponse(answer=answer)


@app.post("/reload-resume")
def reload_resume():
    """Manually re-parse the resume without restarting the server."""
    load_resume()
    if resume_cache is None:
        raise HTTPException(status_code=500, detail="Reload failed. Check server logs.")
    return {"message": "Resume reloaded successfully."}
