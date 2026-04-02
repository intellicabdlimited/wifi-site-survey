from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.data_loader import build_payload, get_options

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/api/options")
def api_options():
    return get_options(str(DATA_DIR))


@app.get("/api/data")
def api_data(metric: str, floor: str, band: str, threshold: float | None = None):
    return build_payload(
        str(DATA_DIR),
        metric=metric,
        floor=floor,
        band=band,
        threshold=threshold,
    )