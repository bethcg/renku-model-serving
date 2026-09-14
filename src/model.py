"""
Model loading for Renku model-serving.

The showcase serves a pretrained Hugging Face model for text classification
(sentiment analysis by default), exactly the way you'd serve any model behind
an inference endpoint.

Configuration (environment variables):
    MODEL_ID   Hugging Face model id   (default: distilbert-base-uncased-finetuned-sst-2-english)
    TASK       transformers pipeline task (default: sentiment-analysis)
    TOP_K      how many labels to return per input (default: 1; set to 0/None for all)

If `transformers` isn't installed yet (e.g. the environment is still building)
or the model can't be loaded, the service falls back to a tiny built-in
keyword classifier so the endpoint is never dead on arrival. The response
shape is identical, so client code doesn't change - only the `backend` field
in /health tells you which one answered.
"""

import os
from typing import Callable, List, Dict

MODEL_ID = os.environ.get("MODEL_ID", "distilbert-base-uncased-finetuned-sst-2-english")
TASK = os.environ.get("TASK", "sentiment-analysis")
_TOP_K_RAW = os.environ.get("TOP_K", "1")
TOP_K = None if _TOP_K_RAW in ("", "0", "None", "none", "all") else int(_TOP_K_RAW)

Prediction = List[Dict[str, float]]  # e.g. [{"label": "POSITIVE", "score": 0.99}]


# --- Fallback: a tiny, dependency-free keyword classifier --------------------
_POSITIVE = {
    "good", "great", "excellent", "amazing", "love", "loved", "wonderful",
    "fantastic", "happy", "best", "awesome", "nice", "brilliant", "perfect",
    "helpful", "clear", "fast", "reliable", "beautiful",
}
_NEGATIVE = {
    "bad", "terrible", "awful", "hate", "hated", "horrible", "worst",
    "broken", "slow", "confusing", "buggy", "useless", "disappointing",
    "poor", "annoying", "crash", "fails", "failed", "ugly",
}


def _fallback_predict(text: str) -> Prediction:
    tokens = [t.strip(".,!?;:()[]\"'").lower() for t in text.split()]
    pos = sum(t in _POSITIVE for t in tokens)
    neg = sum(t in _NEGATIVE for t in tokens)
    if pos == neg:
        label, score = "NEUTRAL", 0.5
    elif pos > neg:
        label, score = "POSITIVE", min(0.5 + 0.1 * (pos - neg), 0.99)
    else:
        label, score = "NEGATIVE", min(0.5 + 0.1 * (neg - pos), 0.99)
    return [{"label": label, "score": round(float(score), 4)}]


def _make_fallback() -> Callable[[str], Prediction]:
    return _fallback_predict


# --- Primary: a real Hugging Face transformers pipeline ----------------------
def _make_transformers_pipeline():
    from transformers import pipeline  # imported lazily so the fallback works

    pipe = pipeline(task=TASK, model=MODEL_ID, top_k=TOP_K)

    def predict(text: str) -> Prediction:
        out = pipe(text)
        # transformers returns either [{label,score}] or [[{...}, ...]]
        if out and isinstance(out[0], list):
            out = out[0]
        return [{"label": r["label"], "score": round(float(r["score"]), 4)} for r in out]

    return predict


def load_predictor():
    """Return (predict_fn, backend_name). Never raises."""
    try:
        predict = _make_transformers_pipeline()
        # warm up / prove it works
        predict("ok")
        return predict, f"transformers:{MODEL_ID}"
    except Exception as exc:  # noqa: BLE001 - we deliberately degrade gracefully
        print(f"[model] transformers backend unavailable ({exc!r}); "
              f"using built-in fallback classifier.", flush=True)
        return _make_fallback(), "fallback:keyword-sentiment"
