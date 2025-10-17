from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from .watermark_removal import WatermarkRemovalError, process_video

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sora 2 Watermark Remover")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/process")
async def process_upload(file: UploadFile = File(...)) -> FileResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    upload_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        output_path = process_video(upload_path, OUTPUT_DIR)
    except WatermarkRemovalError as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def cleanup_files() -> None:
        upload_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{upload_path.stem}_clean.mp4",
        background=BackgroundTask(cleanup_files),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
