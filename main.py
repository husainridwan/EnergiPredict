from __future__ import annotations

import io
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from energipredict import __version__, config as cfg
from energipredict.serve import InvalidUpload, load_artefact, predict_frame

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

MAX_JSON_ROWS = 2000

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _allowed_origins() -> list[str]:
    raw = os.environ.get("ENERGIPREDICT_ALLOWED_ORIGINS", "*")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _metrics_unavailable(message: str) -> dict[str, Any]:
    return {"available": False, "message": message}


def _load_metrics() -> dict[str, Any]:
    if not cfg.METRICS_JSON.exists():
        return _metrics_unavailable(
            "No evaluation results yet. Run `python -m energipredict.train` "
            "to train the models and generate reports/metrics.json."
        )

    try:
        with cfg.METRICS_JSON.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return _metrics_unavailable(
            "Evaluation results are unavailable because reports/metrics.json "
            f"could not be read ({exc}). Run `python -m energipredict.train` to regenerate it."
        )

    if not isinstance(payload, dict):
        return _metrics_unavailable(
            "Evaluation results are unavailable because reports/metrics.json "
            "does not contain a JSON object. Run `python -m energipredict.train` to regenerate it."
        )

    payload["available"] = True
    return payload


def _download_stem(filename: str | None) -> str:
    stem = Path(filename or "").stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-_")
    return stem or "forecast"


STATE: dict[str, Any] = {"metrics": {}, "artefact": None, "artefact_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["metrics"] = _load_metrics()
    try:
        STATE["artefact"] = load_artefact()
    except (FileNotFoundError, ValueError) as exc:
        STATE["artefact"] = None
        STATE["artefact_error"] = str(exc)
    yield
    STATE.clear()


app = FastAPI(
    title="EnergiPredict",
    version=__version__,
    summary="Day-ahead HVAC energy forecasting for large auditoria.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,  # no endpoint authenticates; see module docstring
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _artefact_or_503():
    if STATE.get("artefact") is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "No trained model is available.",
                "remedy": "Run `python -m energipredict.train` to train and save one.",
                "cause": STATE.get("artefact_error"),
            },
        )
    return STATE["artefact"]


# Pages
def _page_context(request: Request) -> dict[str, Any]:
    metrics = STATE.get("metrics") or {}
    artefact = STATE.get("artefact")
    horizons = metrics.get("horizons", {})
    deployed_h = str(metrics.get("deployed_horizon_h", cfg.DEPLOYED_HORIZON_H))
    return {
        "request": request,
        "version": __version__,
        "metrics": metrics,
        "horizons": horizons,
        "deployed_horizon": deployed_h,
        "headline": horizons.get(deployed_h),
        "model": artefact.describe() if artefact else None,
        "target_units": cfg.TARGET_UNITS_SHORT,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Landing page: what the model does, and how well it does it."""
    return templates.TemplateResponse(request, "index.html", _page_context(request))


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Upload page."""
    return templates.TemplateResponse(request, "predict.html", _page_context(request))


@app.get("/benchmarks", response_class=HTMLResponse)
async def benchmarks_page(request: Request):
    """Every model, both horizons, against the naive baselines."""
    return templates.TemplateResponse(request, "benchmarks.html", _page_context(request))


@app.get("/method", response_class=HTMLResponse)
async def method_page(request: Request):
    """How the evaluation is set up, and what the numbers do not cover."""
    return templates.TemplateResponse(request, "method.html", _page_context(request))


# JSON API
@app.get("/health")
async def health():
    """Liveness, plus whether a model is actually loaded."""
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": STATE.get("artefact") is not None,
        "metrics_available": bool((STATE.get("metrics") or {}).get("available")),
    }


@app.get("/api/results")
async def api_results():
    """The full contents of ``reports/metrics.json``.

    Everything here was measured by ``python -m energipredict.train``; no figure
    is transcribed by hand. Safe to embed elsewhere -- it is static JSON and makes
    no model call.
    """
    return STATE.get("metrics") or _load_metrics()


@app.get("/api/model")
async def api_model():
    """Metadata for the deployed model, including what an upload must contain."""
    return _artefact_or_503().describe()


@app.get("/api/sample.csv")
async def api_sample(hours: int = Query(336, ge=48, le=2000)):
    """A ready-made upload, cut from the tail of the training dataset.

    Saves a caller from having to work out the required schema by trial and
    error: download this, upload it, see the shape of a valid request.
    """
    if not cfg.DATA_CSV.exists():
        raise HTTPException(status_code=404, detail="data.csv is not present.")
    wanted = [
        cfg.TIMESTAMP_COL,
        *cfg.HVAC_METER_COLS,
        *cfg.WEATHER_COLS_KEPT,
    ]
    frame = pd.read_csv(cfg.DATA_CSV, usecols=lambda c: c in set(wanted)).tail(hours)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="energipredict-sample.csv"'},
    )


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload, refusing anything over :data:`MAX_UPLOAD_BYTES`.

    Read in chunks and stop at the cap, rather than reading everything and then
    checking the length -- by which point the memory is already committed.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1 << 20):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "File too large.",
                    "limit_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
                },
            )
        chunks.append(chunk)
    if not total:
        raise HTTPException(status_code=422, detail={"error": "The uploaded file is empty."})
    return b"".join(chunks)


def _parse_csv(raw: bytes) -> pd.DataFrame:
    """Parse uploaded bytes as CSV, with messages a caller can act on."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "The file is not UTF-8 text. Export it as CSV UTF-8 "
                "rather than as a spreadsheet or a different encoding.",
                "cause": str(exc),
            },
        ) from exc
    try:
        return pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(
            status_code=422, detail={"error": "The file contains no data rows."}
        ) from exc
    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "The file is not valid CSV.", "cause": str(exc)},
        ) from exc


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(..., description="Hourly CSV history; see /api/model"),
    format: str = Query("json", pattern="^(json|csv)$"),
):
    """Forecast every hour in an uploaded history that has enough context."""
    artefact = _artefact_or_503()
    frame = _parse_csv(await _read_upload(file))

    try:
        predictions, summary = predict_frame(frame, artefact)
    except InvalidUpload as exc:
        raise HTTPException(
            status_code=422, detail={"error": exc.message, **exc.detail}
        ) from exc

    summary["source_filename"] = file.filename

    if format == "csv":
        buffer = io.StringIO()
        predictions.to_csv(buffer)
        stem = _download_stem(file.filename)
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{stem}-forecast.csv"',
                "X-Rows-Predicted": str(summary["rows_predicted"]),
            },
        )

    truncated = len(predictions) > MAX_JSON_ROWS
    body = predictions.head(MAX_JSON_ROWS).reset_index()
    body[cfg.TIMESTAMP_COL] = body[cfg.TIMESTAMP_COL].dt.strftime("%Y-%m-%d %H:%M")
    return JSONResponse(
        {
            "summary": summary,
            "truncated": truncated,
            "rows_returned": len(body),
            "note": (
                f"Showing the first {MAX_JSON_ROWS} of {summary['rows_predicted']} "
                "rows; request format=csv for all of them."
            )
            if truncated
            else None,
            "predictions": body.to_dict(orient="records"),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=bool(os.environ.get("RELOAD")),
    )
