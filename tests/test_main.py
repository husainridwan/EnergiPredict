from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from main import app

ROOT = Path(__file__).resolve().parents[1]


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _valid_csv() -> bytes:
    return pd.read_csv(ROOT / "data.csv").tail(400).to_csv(index=False).encode()


def test_page_routes_render_with_shared_base_template():
    with _client() as client:
        for path in ("/", "/predict", "/benchmarks", "/method"):
            response = client.get(path)
            assert response.status_code == 200
            assert "EnergiPredict" in response.text


def test_read_only_api_routes_are_available():
    with _client() as client:
        health = client.get("/health")
        model = client.get("/api/model")
        results = client.get("/api/results")
        sample = client.get("/api/sample.csv?hours=200")

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert model.status_code == 200
    assert "required_upload_columns" in model.json()
    assert results.status_code == 200
    assert results.json()["available"] is True
    assert sample.status_code == 200
    assert sample.headers["content-type"].startswith("text/csv")


def test_prediction_upload_supports_json_and_csv_download():
    payload = _valid_csv()
    with _client() as client:
        json_response = client.post(
            "/api/predict?format=json",
            files={"file": ("history.csv", payload, "text/csv")},
        )
        csv_response = client.post(
            "/api/predict?format=csv",
            files={"file": ("unsafe name.csv", payload, "text/csv")},
        )

    assert json_response.status_code == 200
    body = json_response.json()
    assert body["summary"]["rows_predicted"] > 0
    assert body["predictions"]
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert "filename=\"unsafe-name-forecast.csv\"" in csv_response.headers["content-disposition"]


def test_empty_upload_gets_an_actionable_validation_error():
    with _client() as client:
        response = client.post(
            "/api/predict?format=json",
            files={"file": ("empty.csv", b"", "text/csv")},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "The uploaded file is empty."
