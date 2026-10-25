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
  /nvidia    — NVIDIA NIM free endpoint (Nemotron 3 Ultra), OpenAI style.

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
     CF_API_TOKEN + CF_ACCOUNT_ID (optional; Cloudflare Workers AI free
       tier), CF_MODELS (comma-separated; default: 4 chat models, first
       success wins). Workers AI answers via its /ai/run/{model} REST
       endpoint; response shape is {"result": {"response": "..."}}.
     NVIDIA_API_KEY (optional), NVIDIA_MODELS (comma-separated;
       default: nvidia/nemotron-3-ultra-550b-a55b — NVIDIA NIM free
       endpoint at integrate.api.nvidia.com/v1, OpenAI style).
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
    _OR_SINGLE or "nvidia/nemotron-3-ultra-550b-a55b:free,"
    "nvidia/nemotron-3-super-120b-a12b:free,"
    "google/gemma-4-31b-it:free,"
    "google/gemma-4-26b-a4b-it:free,"
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
).split(",") if m.strip()]
HF_KEY = os.environ.get("HF_API_KEY", "")
HF_MODELS = [m.strip() for m in os.environ.get(
    "HF_MODELS",
    "mistralai/Mistral-7B-Instruct-v0.3,Qwen/Qwen2.5-7B-Instruct",
).split(",") if m.strip()]
CF_TOKEN = os.environ.get("CF_API_TOKEN", "")
CF_ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "")
CF_MODELS = [m.strip() for m in os.environ.get(
    "CF_MODELS",
    "@cf/meta/llama-3.1-8b-instruct,"
    "@cf/meta/llama-3-8b-instruct,"
    "@cf/mistral/mistral-7b-instruct-v0.1,"
    "@cf/qwen/qwen1.5-7b-chat-awq",
).split(",") if m.strip()]
NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODELS = [m.strip() for m in os.environ.get(
    "NVIDIA_MODELS",
    "nvidia/nemotron-3-ultra-550b-a55b",
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
            "hf_models": HF_MODELS,
            "cf_configured": bool(CF_TOKEN and CF_ACCOUNT),
            "cf_models": CF_MODELS,
            "nvidia_configured": bool(NVIDIA_KEY),
            "nvidia_models": NVIDIA_MODELS}


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


@app.post("/cf")
async def cf(request: Request):
    """Cloudflare Workers AI (free tier), tried in order until one answers.

    REST shape differs from OpenAI style: POST /ai/run/{model} with
    {"messages": [...]}, reply at result.response. Missing token or
    account id -> 503 so the site falls through to built-in.
    """
    if not CF_TOKEN or not CF_ACCOUNT:
        return JSONResponse({"error": "cloudflare not configured"},
                            status_code=503)
    prompt, err = await _prompt(request)
    if err is not None:
        return err
    last = "no models configured"
    for model in CF_MODELS:
        try:
            r = requests.post(
                "https://api.cloudflare.com/client/v4/accounts/"
                + CF_ACCOUNT + "/ai/run/" + model,
                json={"messages": [{"role": "user", "content": prompt}]},
                headers={"Authorization": "Bearer " + CF_TOKEN,
                         "Content-Type": "application/json"},
                timeout=12,
            )
        except requests.RequestException as e:
            last = model + ": unreachable: " + str(e)[:80]
            continue
        try:
            payload = r.json()
        except ValueError:
            last = model + ": non-JSON response"
            continue
        if r.status_code != 200 or not payload.get("success", True):
            errs = payload.get("errors", []) if isinstance(payload, dict) else []
            msg = ""
            try:
                msg = errs[0].get("message", "") if errs else ""
            except (AttributeError, TypeError):
                msg = ""
            last = model + ": HTTP %s %s" % (r.status_code, str(msg)[:100])
            continue
        try:
            text = (payload["result"]["response"] or "").strip()
        except (KeyError, TypeError):
            last = model + ": unexpected shape"
            continue
        if text:
            return JSONResponse({"reply": text, "model": model})
        last = model + ": empty reply"
    return JSONResponse({"error": "all cf models failed: " + last[:200]},
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

@app.post("/nvidia")
async def nvidia(request: Request):
    """NVIDIA NIM free endpoint (Nemotron 3 Ultra), tried in order.

    OpenAI-style chat at integrate.api.nvidia.com/v1. Missing key ->
    503 so the site falls through to its built-in engine.
    """
    if not NVIDIA_KEY:
        return JSONResponse({"error": "nvidia key not configured"},
                            status_code=503)
    prompt, err = await _prompt(request)
    if err is not None:
        return err
    headers = {"Authorization": "Bearer " + NVIDIA_KEY,
               "Content-Type": "application/json"}
    last = "no models configured"
    for model in NVIDIA_MODELS:
        reply, why = _chat(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers, model, prompt, timeout=30)
        if reply is not None:
            return JSONResponse({"reply": reply, "model": model})
        last = model + ": " + why
    return JSONResponse({"error": "all nvidia models failed: " + last[:200]},
                        status_code=502)
