import re
from pathlib import Path

app_path = Path("server/app.py")
code = app_path.read_text(encoding="utf-8")

# 1. Remove the entire markdown renderer
code = re.sub(
    r"# ---------------------------------------------------------------------------\n# Landing-page rendering: pull the PRD-compliant README straight off disk and\n# render it so the HF Spaces \"App\" tab shows the \*real\* card \(with plots\),\n# not the FastAPI scaffold\.\n# ---------------------------------------------------------------------------\n.*?def index\(\) -> HTMLResponse:\n    \"\"\"Render the full PRD-compliant landing page \(README \+ embedded plots\)\.\"\"\"\n    return HTMLResponse\(_render_landing_html\(\)\)\n\n\n",
    """
# ---------------------------------------------------------------------------
# App Initialization & Routing
# ---------------------------------------------------------------------------
""",
    code,
    flags=re.DOTALL
)

# 2. Add StaticFiles import
if "StaticFiles" not in code:
    code = code.replace("from fastapi import FastAPI, HTTPException", "from fastapi import FastAPI, HTTPException\nfrom fastapi.staticfiles import StaticFiles")

# 3. Mount StaticFiles and define new index endpoint
if 'app.mount("/assets"' not in code:
    code = code.replace(
        ")\n\n\n@app.get",
        ")\n\n# Mount the docs/assets folder to serve local graphs\nrepo_root = Path(__file__).parent.parent\napp.mount(\"/assets\", StaticFiles(directory=str(repo_root / \"docs/assets\")), name=\"assets\")\n\n\n@app.get(\"/\", response_class=HTMLResponse)\ndef index() -> HTMLResponse:\n    \"\"\"Return the interactive SPA UI.\"\"\"\n    index_path = Path(__file__).parent / \"index.html\"\n    return HTMLResponse(index_path.read_text(encoding=\"utf-8\"))\n\n\n@app.get"
    )

app_path.write_text(code, encoding="utf-8")
print("Successfully modified app.py")
