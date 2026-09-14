"""
Renku model-serving: a Hugging Face-style inference endpoint.

Served through the same Renku Apps mechanism as any other
app: a Procfile `web` process that listens on $RENKU_SESSION_PORT. Here the web
process is a uvicorn server running this FastAPI app.

Endpoints:
    GET  /            a small "try it" web page
    GET  /health      liveness + which backend answered
    POST /predict     HF Inference API-style: {"inputs": "..."} -> [{label, score}]
    GET  /docs        auto-generated Swagger UI (FastAPI)
"""

import os
from typing import List, Union

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .model import MODEL_ID, TASK, load_predictor

# If Renku serves this session under a sub-path, set RENKU_BASE_URL_PATH (or
# ROOT_PATH) to that path so /docs and the OpenAPI schema resolve correctly.
ROOT_PATH = os.environ.get("RENKU_BASE_URL_PATH", os.environ.get("ROOT_PATH", ""))

app = FastAPI(
    title="Renku model-serving demo",
    description="A Hugging Face-style inference endpoint served on Renku.",
    version="1.0.0",
    root_path=ROOT_PATH,
)

# Load the model once, at startup.
PREDICT, BACKEND = load_predictor()


class InferenceRequest(BaseModel):
    # Mirrors the Hugging Face Inference API request body.
    inputs: Union[str, List[str]]


@app.get("/health")
def health():
    return {"status": "ok", "task": TASK, "model": MODEL_ID, "backend": BACKEND}


@app.post("/predict")
def predict(req: InferenceRequest):
    """HF-style inference. Single string -> one prediction list; list -> many."""
    if isinstance(req.inputs, str):
        return PREDICT(req.inputs)
    return [PREDICT(text) for text in req.inputs]


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Renku model-serving demo</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 system-ui, sans-serif; max-width: 760px; margin: 0 auto;
         padding: 2rem 1rem; }
  h1 { font-size: 1.4rem; margin-bottom: .2rem; }
  .sub { opacity: .7; margin-top: 0; }
  textarea { width: 100%; box-sizing: border-box; padding: .7rem; font: inherit;
             border-radius: 8px; border: 1px solid #8884; min-height: 90px; }
  button { margin-top: .7rem; padding: .55rem 1.1rem; font: inherit; cursor: pointer;
           border-radius: 8px; border: 0; background: #2ca02c; color: #fff; }
  button:disabled { opacity: .6; cursor: default; }
  .card { margin-top: 1.2rem; padding: 1rem; border: 1px solid #8884; border-radius: 10px; }
  .label { font-size: 1.15rem; font-weight: 600; }
  .bar { height: 8px; border-radius: 4px; background: #8883; margin-top: .4rem; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: #2ca02c; }
  pre { background: #8881; padding: .7rem; border-radius: 8px; overflow-x: auto; }
  code { background: #8882; padding: .1rem .3rem; border-radius: 4px; }
  a { color: #2ca02c; }
</style>
</head>
<body>
  <h1>🤗 Renku model-serving demo</h1>
  <p class="sub">A Hugging Face-style inference endpoint, served on Renku via a
     Procfile web process.</p>

  <textarea id="inp" placeholder="Type some text to classify…">This dashboard is fast, clear and genuinely helpful.</textarea>
  <br />
  <button id="go">Run inference</button>

  <div class="card" id="out" hidden>
    <div class="label" id="label">-</div>
    <div class="bar"><i id="fill" style="width:0%"></i></div>
    <p id="score" class="sub"></p>
    <details><summary>Raw JSON</summary><pre id="raw"></pre></details>
  </div>

  <p class="sub" style="margin-top:2rem">
    API: <code>POST ./predict</code> with <code>{"inputs": "your text"}</code> ·
    <a href="./health">/health</a> · <a href="./docs">/docs</a>
  </p>

<script>
const btn = document.getElementById('go');
btn.addEventListener('click', async () => {
  const text = document.getElementById('inp').value;
  btn.disabled = true; btn.textContent = 'Running…';
  try {
    // Relative URL so it works whether served at the root or a sub-path.
    const res = await fetch('./predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs: text })
    });
    const data = await res.json();
    const top = Array.isArray(data) ? data[0] : data;
    document.getElementById('out').hidden = false;
    document.getElementById('label').textContent = top.label;
    document.getElementById('score').textContent = 'confidence ' + (top.score * 100).toFixed(1) + '%';
    document.getElementById('fill').style.width = (top.score * 100) + '%';
    document.getElementById('raw').textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    document.getElementById('out').hidden = false;
    document.getElementById('label').textContent = 'Error';
    document.getElementById('raw').textContent = String(e);
  } finally {
    btn.disabled = false; btn.textContent = 'Run inference';
  }
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML
