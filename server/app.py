"""FastAPI HTTP server exposing DeceptEnv over the OpenEnv contract.

Endpoints
---------
POST /reset       -> start a new episode, returns observation + info
POST /step        -> apply an Agent utterance, returns obs/reward/term/trunc/info
GET  /state       -> full env state (debug; pass `?include_ground_truth=true` for the GT)
GET  /healthz     -> liveness probe (200 OK once the process is up)
GET  /            -> the project's PRD-compliant landing page (rendered from
                    README.md so the HF Spaces "App" tab shows full evidence)
GET  /assets/...  -> static plot files (suspicion / reward / baseline-vs-trained)

Multiple concurrent episodes are supported via the optional `env_id` field on
`/reset` and `/step`. If omitted, a default singleton env (`env_id="main"`) is
used — convenient for curl, vectorisation can pass distinct ids per worker.

Per OpenEnv: this module owns *all* env state. Clients talk to this server
only; they MUST NOT import any other module under `server/`.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from server.env import DeceptEnv, EnvConfig
from server.scenario import list_scenario_ids


# ---------------------------------------------------------------------------
# Landing-page rendering: pull the PRD-compliant README straight off disk and
# render it so the HF Spaces "App" tab shows the *real* card (with plots),
# not the FastAPI scaffold.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_REPO_ROOT_CANDIDATES = [_HERE.parent, _HERE.parent.parent, Path.cwd()]


def _find_repo_file(*relpaths: str) -> Path | None:
    for root in _REPO_ROOT_CANDIDATES:
        for rel in relpaths:
            p = (root / rel).resolve()
            if p.exists():
                return p
    return None


def _embed_image_as_data_uri(path: Path) -> str:
    """Return a data: URI so the image survives cross-origin / asset-routing."""
    try:
        b = path.read_bytes()
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"
    except Exception:
        return ""


def _render_landing_html() -> str:
    readme = _find_repo_file("README.md")
    plots = {
        "baseline_vs_trained.png": _find_repo_file("docs/assets/baseline_vs_trained.png"),
        "suspicion_curve.png":     _find_repo_file("docs/assets/suspicion_curve.png"),
        "reward_curve.png":        _find_repo_file("docs/assets/reward_curve.png"),
    }
    plot_uris = {
        name: (_embed_image_as_data_uri(p) if p else "")
        for name, p in plots.items()
    }

    md_text = readme.read_text(encoding="utf-8") if readme else "# DeceptEnv\n(README.md not found)"
    # Strip the YAML front-matter that's only meaningful to the HF Spaces card.
    if md_text.startswith("---"):
        end = md_text.find("\n---", 3)
        if end != -1:
            md_text = md_text[end + 4 :].lstrip("\n")
    # Replace local plot references with embedded data URIs so they render
    # even when the static asset routes are not served.
    for name, uri in plot_uris.items():
        if uri:
            md_text = md_text.replace(f"docs/assets/{name}", uri)

    body_html = _markdown_to_html(md_text)
    return _PAGE_SHELL.format(body=body_html)


_PAGE_SHELL = """\
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DeceptEnv — Project AI-LIE</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 920px; margin: 2rem auto; padding: 0 1.2rem; line-height: 1.55;
        color: #1f2328; background: #ffffff; }}
 h1 {{ border-bottom: 1px solid #d0d7de; padding-bottom: .3em; margin-top: 1.4em; }}
 h2 {{ border-bottom: 1px solid #d0d7de; padding-bottom: .25em; margin-top: 1.6em; }}
 h3 {{ margin-top: 1.4em; }}
 a  {{ color: #0969da; text-decoration: none; }}
 a:hover {{ text-decoration: underline; }}
 code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.92em; }}
 pre  {{ background: #f6f8fa; padding: 1rem; border-radius: 6px; overflow: auto;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.88em; }}
 pre code {{ background: transparent; padding: 0; }}
 img {{ max-width: 100%; height: auto; display: block; margin: 1rem auto;
       border: 1px solid #d0d7de; border-radius: 6px; }}
 img[src*="img.shields.io"] {{ display: inline-block; margin: 0 8px 0 0; border: none; vertical-align: middle; }}
 table {{ border-collapse: collapse; margin: 1rem 0; }}
 table th, table td {{ border: 1px solid #d0d7de; padding: 6px 12px; text-align: left; }}
 table th {{ background: #f6f8fa; }}
 blockquote {{ margin: 1em 0; padding: .6em 1em; background: #f6f8fa;
              border-left: 4px solid #0969da; border-radius: 4px; color: inherit; }}
 .ribbon {{ background: linear-gradient(90deg,#cc3333 0,#5a3aa6 100%); color: #fff;
           padding: .55rem 1rem; border-radius: 6px; margin-bottom: 1.4rem;
           font-size: .92rem; }}
 .ribbon code {{ background: rgba(255,255,255,.18); color: #fff; }}

 @media (prefers-color-scheme: dark) {{
   body {{ background: #0d1117; color: #e6edf3; }}
   code {{ background: #161b22; color: #e6edf3; }}
   pre  {{ background: #161b22; color: #e6edf3; border: 1px solid #30363d; }}
   table th, table td {{ border-color: #30363d; }}
   table th {{ background: #161b22; }}
   blockquote {{ background: #161b22; border-left-color: #58a6ff; }}
   a {{ color: #58a6ff; }}
 }}
 :root.dark body {{ background: #0d1117; color: #e6edf3; }}
 :root.dark code {{ background: #161b22; color: #e6edf3; }}
 :root.dark pre  {{ background: #161b22; color: #e6edf3; border: 1px solid #30363d; }}
 :root.dark table th, :root.dark table td {{ border-color: #30363d; }}
 :root.dark table th {{ background: #161b22; }}
 :root.dark blockquote {{ background: #161b22; border-left-color: #58a6ff; }}
 :root.dark a {{ color: #58a6ff; }}
</style>
<script>
  if (window.location.search.includes('__theme=dark') || window.matchMedia('(prefers-color-scheme: dark)').matches) {{
    document.documentElement.classList.add('dark');
  }}
</script>
</head><body>
<div class="ribbon">
  <strong>🕵️ DeceptEnv is live</strong> on Hugging Face Spaces — try it now:
  <code>POST /reset</code> · <code>POST /step</code> · <code>GET /state</code> ·
  <a style="color:#fff;text-decoration:underline" href="/healthz">/healthz</a> ·
  <a style="color:#fff;text-decoration:underline" href="/scenarios">/scenarios</a>
</div>
{body}
</body></html>
"""


# ---------------------------------------------------------------------------
# Tiny pure-python Markdown → HTML converter (no third-party dep, keeps the
# Space image small). Handles: headings, lists, tables, fenced code,
# inline code, bold, italic, blockquotes, links, and images.
# ---------------------------------------------------------------------------

import html as _html
import re as _re


def _md_inline(s: str) -> str:
    s = _html.escape(s, quote=False)
    # images: ![alt](url)
    s = _re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',
                r'<img alt="\1" src="\2">', s)
    # links: [text](url)
    s = _re.sub(r'\[([^\]]+)\]\(([^)]*)\)',
                r'<a href="\2">\1</a>', s)
    # inline code: `code`
    s = _re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    # bold: **text**
    s = _re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    # italic: *text* or _text_
    s = _re.sub(r'(?<![*\w])\*([^*\n]+)\*(?!\*)', r'<em>\1</em>', s)
    s = _re.sub(r'(?<![_\w])_([^_\n]+)_(?!_)', r'<em>\1</em>', s)
    return s


def _markdown_to_html(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_list = False
    list_tag = "ul"

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(f"</{list_tag}>")
            in_list = False

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.startswith("```"):
            close_list()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(_html.escape(lines[i], quote=False))
                i += 1
            i += 1  # skip closing ```
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue

        # Heading
        m = _re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            close_list()
            level = len(m.group(1))
            out.append(f"<h{level}>{_md_inline(m.group(2))}</h{level}>")
            i += 1
            continue

        # Horizontal rule
        if _re.match(r"^---+\s*$", line):
            close_list()
            out.append("<hr>")
            i += 1
            continue

        # Blockquote
        if line.startswith("> "):
            close_list()
            buf = []
            while i < len(lines) and lines[i].startswith("> "):
                buf.append(_md_inline(lines[i][2:]))
                i += 1
            out.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue

        # Tables (very simple: header row, separator row, body rows)
        if "|" in line and i + 1 < len(lines) and _re.match(
            r"^\s*\|?[\s:|-]+\|[\s:|-]+\|?\s*$", lines[i + 1]
        ):
            close_list()
            def split_row(r: str) -> list[str]:
                r = r.strip()
                if r.startswith("|"): r = r[1:]
                if r.endswith("|"):  r = r[:-1]
                return [c.strip() for c in r.split("|")]
            headers = split_row(line)
            i += 2  # skip separator
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(split_row(lines[i]))
                i += 1
            t = ["<table><thead><tr>"]
            t += [f"<th>{_md_inline(h)}</th>" for h in headers]
            t += ["</tr></thead><tbody>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue

        # Unordered list
        m = _re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            if not in_list:
                in_list = True; list_tag = "ul"
                out.append("<ul>")
            elif list_tag != "ul":
                out.append(f"</{list_tag}>"); list_tag = "ul"; out.append("<ul>")
            out.append(f"<li>{_md_inline(m.group(1))}</li>")
            i += 1
            continue

        # Ordered list
        m = _re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            if not in_list:
                in_list = True; list_tag = "ol"
                out.append("<ol>")
            elif list_tag != "ol":
                out.append(f"</{list_tag}>"); list_tag = "ol"; out.append("<ol>")
            out.append(f"<li>{_md_inline(m.group(1))}</li>")
            i += 1
            continue

        # Blank line -> paragraph break
        if not line.strip():
            close_list()
            i += 1
            continue

        # Plain paragraph (gather consecutive non-blank lines)
        close_list()
        buf = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not (
            lines[i].startswith("#") or lines[i].startswith("```") or
            lines[i].startswith("> ") or _re.match(r"^\s*[-*]\s+", lines[i]) or
            _re.match(r"^\s*\d+\.\s+", lines[i]) or "|" in lines[i]
        ):
            buf.append(lines[i])
            i += 1
        out.append("<p>" + _md_inline(" ".join(buf)) + "</p>")

    close_list()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    env_id: str = Field("main", description="Identifier for the env session.")
    seed: int | None = Field(None, description="Optional per-episode seed.")
    scenario_id: str | None = Field(
        None, description=f"Optional scenario id; one of {list_scenario_ids()}."
    )


class StepRequest(BaseModel):
    env_id: str = Field("main", description="Identifier for the env session.")
    action: str = Field(..., description="The Agent's natural-language utterance.")


class ResetResponse(BaseModel):
    env_id: str
    observation: dict[str, Any]
    info: dict[str, Any]


class StepResponse(BaseModel):
    env_id: str
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class StateResponse(BaseModel):
    env_id: str
    state: dict[str, Any]


# ---------------------------------------------------------------------------
# Session pool
# ---------------------------------------------------------------------------

class _EnvPool:
    def __init__(self) -> None:
        self._envs: dict[str, DeceptEnv] = {}

    def get_or_create(self, env_id: str) -> DeceptEnv:
        env = self._envs.get(env_id)
        if env is None:
            env = DeceptEnv(config=EnvConfig.from_env())
            self._envs[env_id] = env
        return env

    def get(self, env_id: str) -> DeceptEnv:
        env = self._envs.get(env_id)
        if env is None:
            raise HTTPException(
                status_code=404,
                detail=f"env_id={env_id!r} not found. Call /reset first.",
            )
        return env

    def ids(self) -> list[str]:
        return list(self._envs.keys())


_POOL = _EnvPool()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DeceptEnv",
    version="0.1.0",
    description=(
        "OpenEnv-compliant adversarial-deception environment. The Agent is an "
        "LLM trying to fool a frozen Detective LLM. Reward signal is rubric-"
        "based (suspicion delta + contradiction & evasion penalties)."
    ),
)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Render the full PRD-compliant landing page (README + embedded plots)."""
    return HTMLResponse(_render_landing_html())


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "active_envs": _POOL.ids()}


@app.get("/scenarios")
def scenarios() -> dict[str, Any]:
    return {"scenarios": list_scenario_ids()}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    env = _POOL.get_or_create(req.env_id)
    obs, info = env.reset(seed=req.seed, scenario_id=req.scenario_id)
    return ResetResponse(env_id=req.env_id, observation=obs, info=info)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    env = _POOL.get(req.env_id)
    try:
        obs, reward, terminated, truncated, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(
        env_id=req.env_id,
        observation=obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
def state(env_id: str = "main", include_ground_truth: bool = False) -> StateResponse:
    env = _POOL.get(env_id)
    return StateResponse(env_id=env_id,
                         state=env.state(include_ground_truth=include_ground_truth))


@app.exception_handler(Exception)
async def _unhandled(_, exc: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=500, content={"error": repr(exc)})


# ---------------------------------------------------------------------------
# Entry point: `python -m server.app`
# ---------------------------------------------------------------------------

def _run() -> None:
    import uvicorn
    host = os.environ.get("DECEPTENV_HOST", "0.0.0.0")
    port = int(os.environ.get("DECEPTENV_PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    _run()
