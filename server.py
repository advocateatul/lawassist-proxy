"""
LawAssist Gemini Proxy — 100% FREE (Render free web service).

Why this exists: Google no longer allows "website-restricted" Gemini API
keys (the console forces Gemini keys into service-account "auth keys",
which cannot carry HTTP-referrer restrictions). Embedding an unrestricted
key in the public website repo gets it auto-revoked by Google's leak
scanner. So the key lives HERE, server-side, in an environment variable —
never in any repo, never on the website.

The website POSTs the exact native Gemini request body to /gemini; this
service forwards it to generativelanguage.googleapis.com with the key
attached and returns the native response untouched, so the site's
existing parser works unchanged.

Env: GEMINI_API_KEY (required), GEMINI_MODEL (default gemini-3.8-flash),
     ALLOWED_ORIGINS (comma-separated, optional override).
     OPENROUTER_API_KEY (optional; enables the /openrouter backup route),
     OPENROUTER_MODEL (default deepseek/deepseek-chat-v3-0324:free).

The /openrouter route is the second step in the site's LawAssist chain
(Gemini -> OpenRouter free model -> built-in engine). It accepts a plain
{"prompt": "..."} body and returns {"reply": "..."} in OpenAI style.
Keys stay server-side only, exactly like the Gemini key. Vision
(image) requests are NOT routed here — text only.
"""
import os

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_MODEL = os.environ.get("OPENROUTER_MODEL",
                          "deepseek/deepseek-chat-v3-0324:free")
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


@app.get("/")
def root():
    return {"ok": True, "service": "lawassist-proxy", "model": MODEL,
            "key_configured": bool(KEY),
            "openrouter_configured": bool(OR_KEY),
            "openrouter_model": OR_MODEL}


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


@app.post("/openrouter")
async def openrouter(request: Request):
    """Backup route: plain {"prompt": "..."} -> {"reply": "..."}.

    Text only (no vision). Missing key -> 503 so the site falls through
    to its built-in engine, exactly like a failed Gemini call.
    """
    if not OR_KEY:
        return JSONResponse({"error": "openrouter key not configured"},
                            status_code=503)
    try:
        data = await request.json()
    except ValueError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "empty prompt"}, status_code=400)
    if len(prompt) > 20000:
        return JSONResponse({"error": "prompt too long"}, status_code=413)
    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json={"model": OR_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3,
                  "max_tokens": 2048},
            headers={"Authorization": "Bearer " + OR_KEY,
                     "Content-Type": "application/json",
                     "HTTP-Referer":
                         "https://advocateatul.github.io/advocateatul/",
                     "X-Title": "AK Legal Law - LawAssist"},
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
    if r.status_code != 200:
        err = ""
        try:
            err = payload.get("error", {}).get("message", "")
        except (AttributeError, TypeError):
            err = ""
        return JSONResponse({"error": "openrouter error: " + str(err)[:160]},
                            status_code=r.status_code)
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return JSONResponse({"error": "unexpected upstream shape"},
                            status_code=502)
    return JSONResponse({"reply": text})
