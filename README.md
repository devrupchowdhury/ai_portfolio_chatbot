# Ask My AI — Portfolio Chatbot

A grounded Q&A chatbot for your portfolio site. Visitors (recruiters, HR)
can ask questions and get answers sourced strictly from your resume — no
hallucinated experience or skills.

```
ai-portfolio-chatbot/
├── backend/
│   ├── main.py            FastAPI app: parses resume once, serves /chat
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── ask-my-ai-widget.html   Standalone chat widget (vanilla HTML/CSS/JS)
```

## 1. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: add your GROQ_API_KEY (free at https://console.groq.com/keys)
```

Place your resume PDF in `backend/` and set `RESUME_PATH` in `.env` to its
filename (defaults to `my_resume.pdf`).

Run it:

```bash
uvicorn main:app --reload --port 8000
```

Check it's alive: open `http://localhost:8000/health` — you should see
`"resume_loaded": true`. If it's `false`, check the terminal logs; usually
means the PDF path is wrong or the text extraction came back empty (common
with scanned/image-only PDFs — use a text-based PDF export of your resume).

Test the chat endpoint directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is their strongest technical skill?"}'
```

## 2. Frontend widget

Open `frontend/ask-my-ai-widget.html` directly in a browser to see it working
end-to-end against your local backend. To embed it in your real portfolio:

1. Copy the `<style>` block into your site's CSS (or keep it scoped in a
   `<style>` tag).
2. Copy the `#askmyai-tab` and `#askmyai-panel` markup into your page,
   wherever it should sit in the DOM (it's `position: fixed`, so placement
   in the DOM doesn't matter).
3. Copy the `<script>` block and change `API_BASE_URL` to your deployed
   backend URL.

## 3. Deploying

- **Backend**: any place that runs a long-lived Python process — Render,
  Railway, Fly.io, or a small VPS. Set `GROQ_API_KEY`, `RESUME_PATH`, and
  `ALLOWED_ORIGINS` (your real portfolio domain, not `*`) as environment
  variables on the host.
- **Frontend**: no build step needed — it's static. If your portfolio is
  already a React/Vite app, you can instead port the widget markup into a
  React component and keep the same CSS and fetch logic.

## Notes on the "grounded" part

The system prompt in `ask_candidate()` explicitly instructs the model to
answer only from the parsed resume JSON and to say
*"I don't have enough information to answer that"* when it isn't there.
That's your main lever against hallucination — if you notice it drifting,
tighten that prompt further or add a few worked examples of good "I don't
know" answers.
