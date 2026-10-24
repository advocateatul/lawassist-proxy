"""
LawAssist AI Proxy — 100% FREE (Render free web service).

Why this exists: Google no longer allows "website-restricted" Gemini API
keys (the console forces Gemini keys into service-account "auth keys",
which cannot carry HTTP-referrer restrictions). Embedding an unrestricted
key in the public website repo gets it auto-revoked by Google's leak
scanner. So every key lives HERE, server-side, in environment variables —
never in any repo, never on the website.

Routes (all text-only except /gemini, which also carries vision):
  /gemini     — Gemini (main model), native request body, native reply.
  /groq      — Groq-hosted model (default Llama 3.3), OpenAI style.
  /openrouter — OpenRouter FREE models, tried in order until one answers.
  /hf        — Hugging Face Inference Providers, tried in order.

Every backup route accepts a plain {"prompt": "..."} body and returns
{"reply": "..."}. A missing key returns 503 and a failed provider
returns 502, so the website's chain falls through to the next step and
finally to its built-in engine — never a broken page.

Env: GEMINI_API_KEY (required), GEMINI_MODEL (default gemini-3.8-flash),
     ALLOWED_ORIGINS (comma-separated, optional override).
     GROQ_API_KEY (optional), GROQ_MODEL (default llama-3.3-70b-versatile).
     OPENROUTER_API_KEY (optional),
     OPENROUTER_MODELS (comma-separated; default: 5 free models, first
       success wins; a single OPENROUTER_MODEL overrides the list).
     HF_API_KEY (optional),
     HF_MODELS (comma-separated; default: Mistral-7B-Instruct-v0.3,
       Qwen2.5-7B-Instruct — both openly licensed, no gated approval).
Free-model IDs change over time; a stale ID just fails over to the next
model, and the lists can be refreshed from env with no code change.
"""
import os

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_OR_SINGLE = os.environ.get("OPENROUTER_MODEL", "")
OR_MODELS = [m.strip() for m in os.environ.get(
    "OPENROUTER_MODELS",
    _OR_SINGLE or "deepseek/deepseek-chat-v3-0324:free,"
    "meta-llama/llama-3.3-70b-instruct:free,"
    "google/gemma-3-27b-it:free,"
    "qwen/qwen3-235b-a22b:free,"
    "mistralai/mistral-small-3.1-24b-instruct:free",
).split(",") if m.strip()]
HF_KEY = os.environ.get("HF_API_KEY", "")
HF_MODELS = [m.strip() for m in os.environ.get(
    "HF_MODELS",
    "mistralai/Mistral-7B-Instruct-v0.3,Qwen/Qwen2.5-7B-Instruct",
).split(",") if m.strip()]
ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://advocateatul.github.io,https://advocateatul.eu.org",
).split(",")

app = FastAPI(title="LawAssist Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ORIGINS if o.strip()],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

UPSTREAM = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    + MODEL
    + ":generateContent"
)
MAX_BODY = 12 * 1024 * 1024  # vision requests carry base64 images
SITE_REFERER = "https://advocateatul.github.io/advocateatul/"


@app.get("/")
def root():
    return {"ok": True, "service": "lawassist-proxy", "model": MODEL,
            "key_configured": bool(KEY),
            "groq_configured": bool(GROQ_KEY),
            "groq_model": GROQ_MODEL,
            "openrouter_configured": bool(OR_KEY),
            "openrouter_models": OR_MODELS,
            "hf_configured": bool(HF_KEY),
            "hf_models": HF_MODELS}


async def _prompt(request):
    """Return (prompt, None) or (None, error JSONResponse)."""
    try:
        data = await request.json()
    except ValueError:
        return None, JSONResponse({"error": "invalid JSON body"},
                                  status_code=400)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return None, JSONResponse({"error": "empty prompt"},
                                  status_code=400)
    if len(prompt) > 20000:
        return None, JSONResponse({"error": "prompt too long"},
                                  status_code=413)
    return prompt, None


def _chat(url, headers, model, prompt, timeout):
    """One OpenAI-style chat call. Returns (reply, error-string)."""
    try:
        r = requests.post(
            url,
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3,
                  "max_tokens": 2048},
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as e:
        return None, "unreachable: " + str(e)[:100]
    try:
        payload = r.json()
    except ValueError:
        return None, "non-JSON response"
    if r.status_code != 200:
        err = ""
        try:
            err = payload.get("error", {}).get("message", "")
        except (AttributeError, TypeError):
            err = ""
        return None, "HTTP %s %s" % (r.status_code, str(err)[:120])
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None, "unexpected shape"
    text = (text or "").strip()
    return (text or None), ("" if text else "empty reply")


@app.post("/gemini")
async def gemini(request: Request):
    if not KEY:
        return JSONResponse({"error": "proxy key not configured"},
                            status_code=503)
    body = await request.body()
    if len(body) > MAX_BODY:
        return JSONResponse({"error": "request too large"}, status_code=413)
    try:
        r = requests.post(
            UPSTREAM,
            params={"key": KEY},
            data=body,
            headers={"Content-Type": "application/json"},
            timeout=55,
        )
    except requests.RequestException as e:
        return JSONResponse({"error": "upstream unreachable: " + str(e)[:120]},
                            status_code=502)
    try:
        payload = r.json()
    except ValueError:
        return JSONResponse({"error": "upstream non-JSON response"},
                            status_code=502)
    return JSONResponse(payload, status_code=r.status_code)


@app.post("/groq")
async def groq(request: Request):
    """Groq backup: plain {"prompt"} -> {"reply"}. Missing key -> 503."""
    if not GROQ_KEY:
        return JSONResponse({"error": "groq key not configured"},
                            status_code=503)
    prompt, err = await _prompt(request)
    if err is not None:
        return err
    reply, why = _chat(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": "Bearer " + GROQ_KEY,
         "Content-Type": "application/json"},
        GROQ_MODEL, prompt, timeout=25)
    if reply is None:
        return JSONResponse({"error": "groq error: " + why}, status_code=502)
    return JSONResponse({"reply": reply})


@app.post("/openrouter")
async def openrouter(request: Request):
    """OpenRouter FREE models, tried in order until one answers."""
    if not OR_KEY:
        return JSONResponse({"error": "openrouter key not configured"},
                            status_code=503)
    prompt, err = await _prompt(request)
    if err is not None:
        return err
    headers = {"Authorization": "Bearer " + OR_KEY,
               "Content-Type": "application/json",
               "HTTP-Referer": SITE_REFERER,
               "X-Title": "AK Legal Law - LawAssist"}
    last = "no models configured"
    for model in OR_MODELS:
        reply, why = _chat(
            "https://openrouter.ai/api/v1/chat/completions",
            headers, model, prompt, timeout=10)
        if reply is not None:
            return JSONResponse({"reply": reply, "model": model})
        last = model + ": " + why
    return JSONResponse({"error": "all openrouter models failed: " + last[:200]},
                        status_code=502)


@app.post("/hf")
async def hf(request: Request):
    """Hugging Face Inference Providers, tried in order until one answers."""
    if not HF_KEY:
        return JSONResponse({"error": "hf key not configured"},
                            status_code=503)
    prompt, err = await _prompt(request)
    if err is not None:
        return err
    headers = {"Authorization": "Bearer " + HF_KEY,
               "Content-Type": "application/json"}
    last = "no models configured"
    for model in HF_MODELS:
        reply, why = _chat(
            "https://router.huggingface.co/v1/chat/completions",
            headers, model, prompt, timeout=12)
        if reply is not None:
            return JSONResponse({"reply": reply, "model": model})
        last = model + ": " + why
    return JSONResponse({"error": "all hf models failed: " + last[:200]},
                        status_code=502)
