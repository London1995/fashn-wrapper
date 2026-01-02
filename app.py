import os
import time
import uuid
from pathlib import Path
from typing import Literal, Optional, Dict, Any, Union

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, field_validator

# ---------------------------
# Config
# ---------------------------
FASHN_API_KEY = os.getenv("FASHN_API_KEY")
if not FASHN_API_KEY:
    raise RuntimeError("Missing FASHN_API_KEY env var")

BASE_URL = os.getenv("FASHN_BASE_URL", "https://api.fashn.ai/v1")
HEADERS = {"Authorization": f"Bearer {FASHN_API_KEY}", "Content-Type": "application/json"}

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/tmp/storage")).resolve()

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = (STORAGE_DIR / "models").resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_INDEX_PATH = (STORAGE_DIR / "models.json").resolve()


# If running behind a proxy/Cloudflare, set PUBLIC_BASE_URL to your externally reachable origin
# e.g. https://api.yourdomain.com
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # optional override

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(15 * 1024 * 1024)))  # 15MB default


# ---------------------------
# Presets (lock visual grammar)
# ---------------------------
PRESETS: Dict[str, Dict[str, Any]] = {
    "PDP_STUDIO_SPRING_V1": {
        "tryon_model": "tryon-v1.6",
        "defaults": {},
        "edit_after_tryon": False,
        "edit_defaults": {
            "model": "edit",
            "instruction": "Clean studio lighting, neutral background, preserve garment details, remove artifacts.",
        },
    },
    "PDP_STUDIO_SPRING_V1_EDIT": {
        "tryon_model": "tryon-v1.6",
        "defaults": {},
        "edit_after_tryon": True,
        "edit_defaults": {
            "model": "edit",
            "instruction": "Clean studio lighting, neutral background, preserve garment details, remove artifacts.",
        },
    },
}

# ---------------------------
# App
# ---------------------------
app = FastAPI(title="Thin Wrapper for FASHN (Local Upload MVP)", version="0.2.0")

# Serve uploaded files at /files/<filename>
app.mount("/files", StaticFiles(directory=str(STORAGE_DIR)), name="files")
app.mount("/files/models", StaticFiles(directory=str(MODELS_DIR)), name="models")



# ---------------------------
# Models
# ---------------------------
class UploadResponse(BaseModel):
    file_id: str
    url: HttpUrl
    filename: str
    bytes: int


class ModelRegisterResponse(BaseModel):
    model_id: str
    url: HttpUrl
    filename: str
    bytes: int


class ModelInfo(BaseModel):
    model_id: str
    url: HttpUrl
    filename: str


class RenderPDPRequest(BaseModel):
    # Provide either model_id OR model_image
    model_id: Optional[str] = None
    model_image: Optional[Union[HttpUrl, str]] = None
    garment_image: Union[HttpUrl, str]

    garment_type: Literal["top", "bottom", "full_body"]
    preset: Literal["PDP_STUDIO_SPRING_V1", "PDP_STUDIO_SPRING_V1_EDIT"] = "PDP_STUDIO_SPRING_V1"
    edit_instruction: Optional[str] = None

    timeout_seconds: int = 60
    poll_interval_seconds: float = 1.5

    @field_validator("model_image", "garment_image")
    @classmethod
    def non_empty(cls, v):
        if v is None:
            return v
        if isinstance(v, str) and not v.strip():
            raise ValueError("must not be empty")
        return v
 


class RenderResponse(BaseModel):
    job_id: str
    status: str
    output: Optional[dict] = None


# ---------------------------
# Helpers
# ---------------------------
def public_base(request: Request) -> str:
    """
    Determine base URL to build absolute URLs for /files.
    """
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL.rstrip("/")
    # Use request base url (works locally). Behind a proxy you should set PUBLIC_BASE_URL.
    return str(request.base_url).rstrip("/")


def safe_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return ext if ext in ALLOWED_EXTS else ""


def save_upload(file: UploadFile) -> UploadResponse:
    ext = safe_ext(file.filename or "")
    if not ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTS)}")

    # Enforce size limit by streaming + counting bytes
    file_id = uuid.uuid4().hex
    out_name = f"{file_id}{ext}"
    out_path = STORAGE_DIR / out_name

    total = 0
    with out_path.open("wb") as f:
        while True:
            chunk = file.file.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                out_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_BYTES} bytes.")
            f.write(chunk)

    return UploadResponse(file_id=file_id, url="http://placeholder", filename=out_name, bytes=total)


def resolve_to_url(request: Request, value: Union[HttpUrl, str]) -> str:
    """
    If it's already a URL, return it. If it's a file_id (uuid hex), map to /files/<file>.
    """
    if isinstance(value, HttpUrl):
        return str(value)

    s = value.strip()

    # If user accidentally passes a URL string not parsed as HttpUrl
    if s.startswith("http://") or s.startswith("https://"):
        return s

    # Otherwise treat as file_id and find matching file in storage
    # We accept either:
    # - just file_id (uuid hex)
    # - full stored filename (file_id.ext)
    candidates = []
    if "." in s:
        candidates.append(STORAGE_DIR / s)
    else:
        for ext in ALLOWED_EXTS:
            candidates.append(STORAGE_DIR / f"{s}{ext}")

    found = next((p for p in candidates if p.exists()), None)
    if not found:
        raise HTTPException(status_code=404, detail=f"Local file not found for '{s}'")

    base = public_base(request)
    return f"{base}/files/{found.name}"


def fashn_run(payload: dict) -> str:
    r = requests.post(f"{BASE_URL}/run", headers=HEADERS, json=payload, timeout=30)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail={"msg": "FASHN /run failed", "fashn": r.text})
    data = r.json()
    job_id = data.get("id") or data.get("prediction_id") or data.get("job_id")
    if not job_id:
        raise HTTPException(status_code=502, detail={"msg": "No job id returned by FASHN", "fashn": data})
    return job_id


def fashn_poll(job_id: str, timeout_seconds: int, poll_interval: float) -> dict:
    deadline = time.time() + timeout_seconds
    last = None
    while time.time() < deadline:
        r = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=30)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail={"msg": "FASHN /status failed", "fashn": r.text})
        last = r.json()
        status = (last.get("status") or "").lower()
        if status in {"completed", "succeeded", "success", "done"}:
            return last
        if status in {"failed", "error", "canceled", "cancelled"}:
            raise HTTPException(status_code=502, detail={"msg": "FASHN job failed", "fashn": last})
        time.sleep(poll_interval)

    raise HTTPException(status_code=504, detail={"msg": "Timed out waiting for FASHN job", "last": last})


def pick_first_image_url(output: Any) -> Optional[str]:
    """
    Best-effort extraction of an image URL from output payloads.
    (FASHN output formats can vary by model/config.)
    """
    if isinstance(output, dict):
        if isinstance(output.get("image_url"), str):
            return output["image_url"]
        imgs = output.get("images")
        if isinstance(imgs, list) and imgs and isinstance(imgs[0], str):
            return imgs[0]
    return None

def load_models_index() -> Dict[str, Dict[str, Any]]:
    if MODELS_INDEX_PATH.exists():
        import json
        try:
            return json.loads(MODELS_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_models_index(index: Dict[str, Dict[str, Any]]) -> None:
    import json
    MODELS_INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def next_model_id(index: Dict[str, Dict[str, Any]]) -> str:
    nums = []
    for k in index.keys():
        if isinstance(k, str) and k.startswith("M") and k[1:].isdigit():
            nums.append(int(k[1:]))
    n = (max(nums) + 1) if nums else 1
    return f"M{n:02d}"


def save_model_upload(file: UploadFile) -> ModelRegisterResponse:
    ext = safe_ext(file.filename or "")
    if not ext:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTS)}")

    index = load_models_index()
    model_id = next_model_id(index)

    out_name = f"{model_id}{ext}"
    out_path = MODELS_DIR / out_name

    total = 0
    with out_path.open("wb") as f:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                out_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_BYTES} bytes.")
            f.write(chunk)

    index[model_id] = {"filename": out_name, "bytes": total}
    save_models_index(index)

    return ModelRegisterResponse(model_id=model_id, url="http://placeholder", filename=out_name, bytes=total)


def resolve_model_id_to_url(request: Request, model_id: str) -> str:
    index = load_models_index()
    meta = index.get(model_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Unknown model_id '{model_id}'")
    filename = meta.get("filename")
    if not filename:
        raise HTTPException(status_code=404, detail=f"Model file missing for '{model_id}'")

    path = MODELS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found on disk for '{model_id}'")

    base = public_base(request)
    return f"{base}/files/models/{path.name}"


# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True, "storage_dir": str(STORAGE_DIR)}


@app.post("/upload", response_model=UploadResponse)
def upload_image(request: Request, file: UploadFile = File(...)):
    saved = save_upload(file)
    base = public_base(request)
    saved.url = f"{base}/files/{saved.filename}"
    return saved
@app.post("/models/register", response_model=ModelRegisterResponse)
def register_model(request: Request, file: UploadFile = File(...)):
    saved = save_model_upload(file)
    base = public_base(request)
    saved.url = f"{base}/files/models/{saved.filename}"
    return saved


@app.get("/models", response_model=list[ModelInfo])
def list_models(request: Request):
    base = public_base(request)
    index = load_models_index()
    out: list[ModelInfo] = []
    for model_id, meta in sorted(index.items()):
        filename = meta.get("filename")
        if not filename:
            continue
        out.append(ModelInfo(model_id=model_id, filename=filename, url=f"{base}/files/models/{filename}"))
    return out



@app.post("/render/pdp", response_model=RenderResponse)
def render_pdp(request: Request, req: RenderPDPRequest):
    preset = PRESETS.get(req.preset)
    if not preset:
        raise HTTPException(status_code=400, detail="Unknown preset")

    if req.model_id:
        model_url = resolve_model_id_to_url(request, req.model_id)
    elif req.model_image is not None:
        model_url = resolve_to_url(request, req.model_image)
    else:
        raise HTTPException(status_code=400, detail="Provide either model_id or model_image")

    garment_url = resolve_to_url(request, req.garment_image)


    # 1) Try-on
    tryon_payload = {
        "model": preset["tryon_model"],
        "input": {
            "image_model_url": model_url,
            "image_garment_url": garment_url,
            "garment_type": req.garment_type,
            **preset.get("defaults", {}),
        },
    }

    tryon_job_id = fashn_run(tryon_payload)
    tryon_result = fashn_poll(tryon_job_id, req.timeout_seconds, req.poll_interval_seconds)
    output = tryon_result.get("output") or tryon_result

    # 2) Optional edit pass
    if preset.get("edit_after_tryon"):
        instruction = req.edit_instruction or preset["edit_defaults"]["instruction"]
        input_image_url = pick_first_image_url(output) or model_url

        edit_payload = {
            "model": preset["edit_defaults"]["model"],
            "input": {
                "image_url": input_image_url,
                "instruction": instruction,
            },
        }
        edit_job_id = fashn_run(edit_payload)
        edit_result = fashn_poll(edit_job_id, req.timeout_seconds, req.poll_interval_seconds)
        return RenderResponse(job_id=edit_job_id, status="completed", output=edit_result.get("output") or edit_result)

    return RenderResponse(job_id=tryon_job_id, status="completed", output=output)

