from pathlib import Path

with open("server/app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Drop the legacy markdown parser (lines 35 to 321 inclusive)
new_lines = lines[:34] + lines[321:]
source = "".join(new_lines)

# 1. Add StaticFiles import
source = source.replace("from fastapi import FastAPI, HTTPException", "from fastapi import FastAPI, HTTPException\nfrom fastapi.staticfiles import StaticFiles")

# 2. Add Mount
source = source.replace(
    ")\n\n\n@app.get",
    ")\n\nrepo_root = Path(__file__).resolve().parent.parent\napp.mount(\"/assets\", StaticFiles(directory=str(repo_root / \"docs\" / \"assets\")), name=\"assets\")\n\n\n@app.get"
)

# 3. Replace index endpoint
old_index = """@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    \"\"\"Render the full PRD-compliant landing page (README + embedded plots).\"\"\"
    return HTMLResponse(_render_landing_html())"""

new_index = """@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    \"\"\"Return the interactive SPA UI.\"\"\"
    index_path = Path(__file__).parent / \"index.html\"
    return HTMLResponse(index_path.read_text(encoding=\"utf-8\"))"""

source = source.replace(old_index, new_index)

with open("server/app.py", "w", encoding="utf-8") as f:
    f.write(source)
print("done")
