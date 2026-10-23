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
"""
import os

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
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
            "key_configured": bool(KEY)}


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
