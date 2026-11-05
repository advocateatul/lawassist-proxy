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
  /websearch — live web search for LawAssist when AI + built-in both\n              have no answer. Brave Search API (BRAVE_API_KEY env) first,\n              DuckDuckGo Instant Answer API (no key) as fallback. Returns\n              titles, snippets and URLs. Rate-limited per IP (20/min, 100/hour).

Every backup route accepts a plain {"prompt": "..."} body and returns
{"reply": "..."}. A missing key returns 503 and a failed provider
returns 502, so the website's chain falls through to the next step and
finally to its built-in engine — never a broken page.

Env: GEMINI_API_KEY (required), GEMINI_MODEL (default gemini-3.8-flash),
     ALLOWED_ORIGINS (comma-separated, optional override).
     GROQ_API_KEY (optional), GROQ_MODELS (default gpt-oss-120b first; llama-3.3-70b-versatile retired by Groq Aug 2026).
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
import hashlib
import hmac
import os
import re
from urllib.parse import parse_qs, unquote, urlparse

import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
_GROQ_SINGLE = os.environ.get("GROQ_MODEL", "")
GROQ_MODELS = [m.strip() for m in os.environ.get(
    "GROQ_MODELS",
    _GROQ_SINGLE or "openai/gpt-oss-120b,"
    "openai/gpt-oss-20b,"
    "meta-llama/llama-4-scout-17b-16e-instruct,"
    "meta-llama/llama-4-maverick-17b-128e-instruct",
).split(",") if m.strip()]
GROQ_MODEL = GROQ_MODELS[0] if GROQ_MODELS else ""
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_OR_SINGLE = os.environ.get("OPENROUTER_MODEL", "")
OR_MODELS = [m.strip() for m in os.environ.get(
    "OPENROUTER_MODELS",
    _OR_SINGLE or "nvidia/nemotron-3-ultra-550b-a55b:free,"
    "nvidia/nemotron-3-super-120b-a12b:free,"
    "google/gemma-4-31b-it:free,"
    "google/gemma-4-26b-a4b-it:free,"
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,"
    "nvidia/nemotron-3.5-lightning:free,"
    "thinkingmachines/inkling:free,"
    "xingchen-agi/xing4.0-29b-a4b:free,"
  "recursive/flint-v1.0:free,"
  "nvidia/nemotron-3.5-super-vl:free,"
  "thinkingmachines/inkling-small:free",
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
    "@cf/qwen/qwen1.5-7b-chat-awq,"
  "@cf/qwen/qwen3-30b-a3b-fp8,"
  "@cf/deepseek-ai/deepseek-v4.1-flash-1017,"
  "@cf/moonshotai/kimi-k2.6,"
  "@cf/nvidia/nemotron-3-120b-a12b,"
  "@cf/qwen/qwen3.8-27b"
).split(",") if m.strip()]
NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_MODELS = [m.strip() for m in os.environ.get(
    "NVIDIA_MODELS",
    "nvidia/nemotron-3-ultra-550b-a55b,"
    "nvidia/nemotron-3-super-120b-a12b,"
    "nvidia/nemotron-3-nano-30b-a3b,"
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
).split(",") if m.strip()]
SAMBANOVA_KEY = os.environ.get("SAMBANOVA_API_KEY", "")
SAMBANOVA_MODELS = [m.strip() for m in os.environ.get(
    "SAMBANOVA_MODELS",
    "Meta-Llama-3.3-70B-Instruct,DeepSeek-V3.1,gpt-oss-120b",
).split(",") if m.strip()]
LLM7_KEY = os.environ.get("LLM7_API_KEY", "")
LLM7_MODELS = [m.strip() for m in os.environ.get(
    "LLM7_MODELS",
    "nemotron-3-nano:30b,mistral-Nemo-Instruct-2407,"
    "DeepSeek-V4-Flash-0731,GLM-5.3-Flash",
).split(",") if m.strip()]
KILO_KEY = os.environ.get("KILO_API_KEY", "")
KILO_MODELS = [m.strip() for m in os.environ.get(
    "KILO_MODELS",
    "nvidia/nemotron-3.5-super-vl:free",
).split(",") if m.strip()]
POLLINATIONS_KEY = os.environ.get("POLLINATIONS_API_KEY", "")
POLLINATIONS_MODELS = [m.strip() for m in os.environ.get(
    "POLLINATIONS_MODELS",
    "gpt-oss-20b",
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
# --- Per-IP abuse guard (2026-11-04 security audit) ---
# This proxy has no login, and CORS alone cannot stop direct script/curl
# calls, so anyone could otherwise hammer the free-model quotas and starve
# real visitors. Bounds: 30 AI POSTs / 60s and 300 / 3600s per caller IP
# (X-Forwarded-For first entry, else peer). Human use never reaches these
# numbers (one question = at most ~10 route attempts). Fail-open: any error
# inside the guard lets the request through, so a guard bug can never take
# LawAssist down. GET / (health) is exempt.
import time as _time
_RATE_WINDOWS = (("60s", 60, 30), ("3600s", 3600, 300))
_RATE_BUCKETS = {}
def _caller_ip(request):
    try:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
    except Exception:
        pass
    try:
        return request.client.host if request.client else "unknown"
    except Exception:
        return "unknown"
@app.middleware("http")
async def _rate_guard(request, call_next):
    try:
        if request.method == "POST" and request.url.path != "/":
            ip = _caller_ip(request)
            now = _time.time()
            for _name, _window, _cap in _RATE_WINDOWS:
                key = (ip, _name)
                start, count = _RATE_BUCKETS.get(key, (now, 0))
                if now - start >= _window:
                    start, count = now, 0
                count += 1
                _RATE_BUCKETS[key] = (start, count)
                if count > _cap:
                    return JSONResponse(
                        {"error": "rate limited, try again shortly"},
                        status_code=429, headers={"Retry-After": "60"})
            if len(_RATE_BUCKETS) > 10000:
                _RATE_BUCKETS.clear()
    except Exception:
        pass
    return await call_next(request)


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
            "groq_models": GROQ_MODELS,
            "openrouter_configured": bool(OR_KEY),
            "openrouter_models": OR_MODELS,
            "hf_configured": bool(HF_KEY),
            "hf_models": HF_MODELS,
            "cf_configured": bool(CF_TOKEN and CF_ACCOUNT),
            "cf_models": CF_MODELS,
            "nvidia_configured": bool(NVIDIA_KEY),
            "nvidia_models": NVIDIA_MODELS,
            "sambanova_configured": bool(SAMBANOVA_KEY),
            "sambanova_models": SAMBANOVA_MODELS,
            "llm7_configured": bool(LLM7_KEY),
            "llm7_models": LLM7_MODELS,
            "kilo_configured": bool(KILO_KEY),
            "kilo_models": KILO_MODELS,
            "pollinations_configured": bool(POLLINATIONS_KEY),
            "pollinations_models": POLLINATIONS_MODELS}


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


def _chat(url, headers, model, prompt, timeout, max_tokens=2048, extra=None):
    """One OpenAI-style chat call. Returns (reply, error-string).

    Only the message "content" is ever returned — "reasoning" and
    "reasoning_content" fields are never passed through, so a model's
    chain-of-thought cannot leak into the site's answer.
    """
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens}
    if extra:
        body.update(extra)
    try:
        r = requests.post(
            url,
            json=body,
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
        msg = payload["choices"][0]["message"]
        text = msg.get("content")  # content only — never reasoning fields
    except (KeyError, IndexError, TypeError, AttributeError):
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
    reply, why, used = None, "no models configured", ""
    for _m in GROQ_MODELS:
        reply, why = _chat(
            "https://api.groq.com/openai/v1/chat/completions",
            {"Authorization": "Bearer " + GROQ_KEY,
             "Content-Type": "application/json"},
            _m, prompt, timeout=25)
        if reply is not None:
            used = _m
            break
    if reply is None:
        return JSONResponse({"error": "groq error: " + why}, status_code=502)
    return JSONResponse({"reply": reply, "model": used})


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
    only = request.query_params.get("model")
    try_models = [only] if only else OR_MODELS
    for model in try_models:
        reply, why = _chat(
            "https://openrouter.ai/api/v1/chat/completions",
            headers, model, prompt, timeout=10, max_tokens=4096,
            extra={"reasoning": {"exclude": True}})
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
    tried_errors = []
    only = request.query_params.get("model")
    try_models = [only] if only else CF_MODELS
    for model in try_models:
        try:
            body = {"messages": [{"role": "user", "content": prompt}]}
            if model == "@cf/qwen/qwen3.8-27b":
                body["reasoning_effort"] = "low"
            r = requests.post(
                "https://api.cloudflare.com/client/v4/accounts/"
                + CF_ACCOUNT + "/ai/run/" + model,
                json=body,
                headers={"Authorization": "Bearer " + CF_TOKEN,
                         "Content-Type": "application/json"},
                timeout=60
            )
        except requests.RequestException as e:
            last = model + ": unreachable: " + str(e)[:80]
            tried_errors.append(last)
            continue
        try:
            payload = r.json()
        except ValueError:
            last = model + ": non-JSON response"
            tried_errors.append(last)
            continue
        if r.status_code != 200 or not payload.get("success", True):
            errs = payload.get("errors", []) if isinstance(payload, dict) else []
            msg = ""
            try:
                msg = errs[0].get("message", "") if errs else ""
            except (AttributeError, TypeError):
                msg = ""
            last = model + ": HTTP %s %s" % (r.status_code, str(msg)[:100])
            tried_errors.append(last)
            continue
        try:
            res = payload["result"]
            if isinstance(res, str):
                text = res.strip()
            elif isinstance(res, dict) and isinstance(res.get("response"), str):
                text = (res["response"] or "").strip()
            elif isinstance(res, dict) and res.get("choices"):
                ch = res["choices"][0]
                msg = ch.get("message") if isinstance(ch, dict) else None
                if isinstance(msg, dict):
                    text = (msg.get("content") or "").strip()
                elif isinstance(ch, dict):
                    text = (ch.get("text") or "").strip()
                else:
                    text = ""
            else:
                raise KeyError("no response or choices")
        except (KeyError, TypeError, IndexError):
            res = payload.get("result") if isinstance(payload, dict) else None
            keys = list(res.keys()) if isinstance(res, dict) else type(res).__name__
            last = model + ": unexpected shape result_keys=" + str(keys)[:120]
            tried_errors.append(last)
            continue
        if text:
            if request.query_params.get("debug") == "1":
                return JSONResponse({"reply": text, "model": model, "tried_errors": tried_errors})
            return JSONResponse({"reply": text, "model": model})
        last = model + ": empty reply"
        tried_errors.append(last)
    return JSONResponse({"error": "all cf models failed: " + last[:200], "tried_errors": tried_errors},
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

@app.post("/sambanova")
async def sambanova(request: Request):
    """SambaNova Cloud free tier, OpenAI style at api.sambanova.ai/v1. Missing key -> 503."""
    if not SAMBANOVA_KEY:
        return JSONResponse({"error": "sambanova key not configured"},
                            status_code=503)
    prompt, err = await _prompt(request)
    if err is not None:
        return err
    headers = {"Authorization": "Bearer " + SAMBANOVA_KEY,
               "Content-Type": "application/json"}
    last = "no models configured"
    only = request.query_params.get("model")
    try_models = [only] if only else SAMBANOVA_MODELS
    for model in try_models:
        reply, why = _chat(
            "https://api.sambanova.ai/v1/chat/completions",
            headers, model, prompt, timeout=30)
        if reply is not None:
            return JSONResponse({"reply": reply, "model": model})
        last = model + ": " + why
    return JSONResponse({"error": "all sambanova models failed: " + last[:200]},
                        status_code=502)

@app.post("/llm7")
async def llm7(request: Request):
    """LLM7.io free token, OpenAI style at api.llm7.io/v1. Missing key -> 503."""
    if not LLM7_KEY:
        return JSONResponse({"error": "llm7 key not configured"},
                            status_code=503)
    prompt, err = await _prompt(request)
    if err is not None:
        return err
    headers = {"Authorization": "Bearer " + LLM7_KEY,
               "Content-Type": "application/json"}
    last = "no models configured"
    only = request.query_params.get("model")
    try_models = [only] if only else LLM7_MODELS
    for model in try_models:
        reply, why = _chat(
            "https://api.llm7.io/v1/chat/completions",
            headers, model, prompt, timeout=30)
        if reply is not None:
            return JSONResponse({"reply": reply, "model": model})
        last = model + ": " + why
    return JSONResponse({"error": "all llm7 models failed: " + last[:200]},
                        status_code=502)

@app.post("/kilo")
async def kilo(request: Request):
    """Kilo Gateway free models, OpenAI style at ai-gateway.kilo.ai/api/v1. Missing key -> 503."""
    if not KILO_KEY:
        return JSONResponse({"error": "kilo key not configured"},
                            status_code=503)
    prompt, err = await _prompt(request)
    if err is not None:
        return err
    headers = {"Authorization": "Bearer " + KILO_KEY,
               "Content-Type": "application/json"}
    last = "no models configured"
    only = request.query_params.get("model")
    try_models = [only] if only else KILO_MODELS
    for model in try_models:
        reply, why = _chat(
            "https://ai-gateway.kilo.ai/api/v1/chat/completions",
            headers, model, prompt, timeout=30)
        if reply is not None:
            return JSONResponse({"reply": reply, "model": model})
        last = model + ": " + why
    return JSONResponse({"error": "all kilo models failed: " + last[:200]},
                        status_code=502)

@app.post("/pollinations")
async def pollinations(request: Request):
    """Pollinations registered key, OpenAI style at gen.pollinations.ai/v1. Missing key -> 503."""
    if not POLLINATIONS_KEY:
        return JSONResponse({"error": "pollinations key not configured"},
                            status_code=503)
    prompt, err = await _prompt(request)
    if err is not None:
        return err
    headers = {"Authorization": "Bearer " + POLLINATIONS_KEY,
               "Content-Type": "application/json"}
    last = "no models configured"
    only = request.query_params.get("model")
    try_models = [only] if only else POLLINATIONS_MODELS
    for model in try_models:
        reply, why = _chat(
            "https://gen.pollinations.ai/v1/chat/completions",
            headers, model, prompt, timeout=30)
        if reply is not None:
            return JSONResponse({"reply": reply, "model": model})
        last = model + ": " + why
    return JSONResponse({"error": "all pollinations models failed: " + last[:200]},
                        status_code=502)

# --- Documents gate (security-audit fix) ---
# The site's Documents password is verified here, never in the public file.
# DOCS_PASSWORD_HASH = SHA-256 hex of the password (Render env only).
# Folders come from DOCS_LABEL_N / DOCS_EMBED_N / DOCS_URL_N env vars.
# The per-IP guard above already caps brute force (30/min, 300/hour).
DOCS_HASH = os.environ.get("DOCS_PASSWORD_HASH", "").strip().lower()
def _docs_folders():
    out = []
    for i in (1, 2, 3):
        label = os.environ.get(f"DOCS_LABEL_{i}", "")
        embed = os.environ.get(f"DOCS_EMBED_{i}", "")
        url = os.environ.get(f"DOCS_URL_{i}", "")
        if label and embed and url:
            out.append({"label": label, "embedUrl": embed, "folderUrl": url})
    return out
@app.post("/docs")
async def docs_gate(request: Request):
    try:
        data = await request.json()
    except ValueError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    password = data.get("password") or ""
    if not isinstance(password, str):
        return JSONResponse({"error": "invalid password"}, status_code=400)
    ok = bool(DOCS_HASH) and hmac.compare_digest(
        hashlib.sha256(password.encode("utf-8")).hexdigest(), DOCS_HASH)
    if not ok:
        return JSONResponse({"error": "incorrect password"}, status_code=403)
    folders = _docs_folders()
    if not folders:
        return JSONResponse({"error": "documents not configured"}, status_code=503)
    return {"folders": folders}


# ---- /websearch (2026-11-05): live web search for LawAssist ----
# When the site's AI chain + built-in engine both have no answer, the site
# asks here for live internet results. Two sources, in order:
#   1. Brave Search API (BRAVE_API_KEY in Render env; free tier) — proper
#      web results incl. law sites and news. Key is created by Atul in his
#      Brave account and stored as an env var only, never in chat/repo.
#   2. DuckDuckGo Instant Answer API (public, no key) — short abstracts
#      only (mostly Wikipedia), used when no Brave key is configured.
# NOTE (2026-11-05, tested): DuckDuckGo/Mojeek HTML scraping is DEAD from
# datacenter IPs — both serve bot-challenge/403 pages to this VM and would
# do the same from Render. Do NOT re-add HTML scraping as a source.
_WS_WINDOWS = (("60s", 60, 20), ("3600s", 3600, 100))
_WS_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
BRAVE_KEY = os.environ.get("BRAVE_API_KEY", "")

def _ws_rate_limited(request):
    try:
        ip = _caller_ip(request)
        now = _time.time()
        for _name, _window, _cap in _WS_WINDOWS:
            key = ("ws", ip, _name)
            start, count = _RATE_BUCKETS.get(key, (now, 0))
            if now - start >= _window:
                start, count = now, 0
            count += 1
            _RATE_BUCKETS[key] = (start, count)
            if count > _cap:
                return True
    except Exception:
        return False
    return False

def _ws_brave(q):
    if not BRAVE_KEY:
        return []
    try:
        r = requests.get("https://api.search.brave.com/res/v1/web/search",
                         params={"q": q, "count": 5},
                         headers={"Accept": "application/json",
                                  "Accept-Encoding": "gzip",
                                  "X-Subscription-Token": BRAVE_KEY},
                         timeout=10)
        if r.status_code != 200:
            return []
        out = []
        for item in (r.json().get("web") or {}).get("results") or []:
            url = (item.get("url") or "").strip()
            if not url.startswith("http"):
                continue
            out.append({"title": (item.get("title") or "").strip(),
                        "url": url,
                        "snippet": (item.get("description") or "").strip()})
        return out[:5]
    except Exception:
        return []

def _ws_ddg_instant(q):
    try:
        r = requests.get("https://api.duckduckgo.com/",
                         params={"q": q, "format": "json", "no_html": "1",
                                 "skip_disambig": "1"},
                         headers=_WS_UA, timeout=10)
        if r.status_code != 200:
            return []
        j = r.json()
        out = []
        if j.get("AbstractText") and j.get("AbstractURL"):
            out.append({"title": (j.get("Heading") or j.get("AbstractSource") or "Web result").strip(),
                        "url": j["AbstractURL"],
                        "snippet": j["AbstractText"].strip()})
        for t in (j.get("RelatedTopics") or [])[:4]:
            if isinstance(t, dict) and t.get("Text") and t.get("FirstURL"):
                out.append({"title": (t.get("Text") or "")[:80].strip(),
                            "url": t["FirstURL"],
                            "snippet": (t.get("Text") or "").strip()})
        return out[:5]
    except Exception:
        return []

@app.get("/websearch")
async def websearch(request: Request, q: str = ""):
    q = (q or "").strip()
    if not q or len(q) > 200:
        return JSONResponse({"error": "bad query"}, status_code=400)
    if _ws_rate_limited(request):
        return JSONResponse({"error": "rate limited, try again shortly"},
                            status_code=429, headers={"Retry-After": "60"})
    results = _ws_brave(q)
    if not results:
        results = _ws_ddg_instant(q)
    if not results:
        return JSONResponse({"error": "search unavailable", "results": []},
                            status_code=502)
    return {"results": results}
