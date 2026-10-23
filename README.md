# LawAssist Gemini Proxy (Render free tier)

Purpose: hold the Gemini API key server-side (env var) and forward the
website's native Gemini requests. The key never appears in any repo or
on the website — Google no longer offers website-restricted Gemini keys,
and unrestricted embedded keys get auto-revoked by Google's leak scanner.

Verified locally 2026-10-23: health endpoint OK; no-key → clean 503;
fake key → Google's real 400 API_KEY_INVALID passed through verbatim
(forwarding path works end-to-end).

## Deploy (one time)
1. Create a GitHub repo `lawassist-proxy` (public code is fine — no
   secrets in it) containing server.py, requirements.txt, render.yaml.
2. Render.com → sign up with GitHub (free Hobby workspace, no card).
3. New → Web Service → connect the repo (render.yaml is picked up).
4. Environment: set GEMINI_API_KEY to the AI Studio "LawAssist" key
   (paste from AI Studio — the key's only copy outside the vault).
5. Deploy → note the service URL, e.g. https://lawassist-proxy.onrender.com

## Wire the website (after deploy)
- In index.html LawAssist code: when GEMINI_KEY is empty, POST the same
  native body to <proxy>/gemini instead of generativelanguage directly.
  Response shape is identical (native passthrough).
- Free tier sleeps after 15 min idle (~1 min wake). Keep-alive: ping
  GET / every ~10 min during the day (750 free instance-hours/month).

## Later
- When the Oracle server exists, this proxy is replaced by the
  self-hosted backend and the Render service can be deleted.
