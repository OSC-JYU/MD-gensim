import asyncio
import contextlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import PlainTextResponse

import storage
import bow
import similarity
import topics
from md_service_boilerplate import create_md_service_app

# DEBUG=1 logs every incoming message and every response in full.
DEBUG = os.environ.get("DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
logging.basicConfig(level="DEBUG" if DEBUG else os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("md-gensim")

HERE = Path(__file__).resolve().parent
MODE = storage.storage_mode()
# /config reports the adapter that matches the storage mode, so the consumer picks the right one.
os.environ.setdefault("SERVICE_ADAPTER", storage.adapter_for(MODE))
DESCRIPTOR = Path(os.environ.get("SERVICE_DESCRIPTOR_PATH", HERE / "service.json")).resolve()


def dispatch(message: dict, store) -> dict:
    task_id = (message.get("task") or {}).get("id")
    if task_id == "bow":
        return bow.run_bow(message, store)
    if task_id in ("similarity_index", "similarity_index_set"):
        return similarity.build_index(message, store)
    if task_id == "search":
        return similarity.search(message, store)
    if task_id in ("topics", "topics_text"):
        return topics.run_topics(message, store)
    raise HTTPException(status_code=400, detail=f"Unknown task: {task_id}")


def describe(message: dict) -> str:
    task = message.get("task") or {}
    file = message.get("file") or {}
    model = task.get("model")
    model_id = model.get("id") if isinstance(model, dict) else model
    parts = [f"task={task.get('id')}", f"model={model_id}", f"file={file.get('@rid')} {file.get('path')}"]
    if message.get("files"):
        parts.append(f"files={len(message['files'])}")
    return " ".join(parts)


def run_task(message: dict, store) -> dict:
    """Runs one task and logs why it failed; the reason also goes back to the adapter in `detail`."""
    started = time.monotonic()
    logger.info("start %s (%s mode)", describe(message), MODE)
    if DEBUG:
        shown = {**message, "files": f"[{len(message['files'])} files]"} if message.get("files") else message
        logger.debug("message: %s", json.dumps(shown, ensure_ascii=False, default=str)[:20000])
    try:
        result = dispatch(message, store)
    except HTTPException as error:
        logger.error("failed %s: %s %s", describe(message), error.status_code, error.detail)
        raise
    except Exception as error:
        logger.exception("crashed %s", describe(message))
        raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}") from error
    logger.info("done %s in %.2fs", describe(message), time.monotonic() - started)
    if DEBUG:
        logger.debug("response: %s", json.dumps(result, ensure_ascii=False)[:5000])
    return result


def save_upload(upload, store_dir: Path) -> Path:
    suffix = Path(getattr(upload, "filename", "") or "").suffix
    target = store_dir / f"in_{uuid4().hex}{suffix}"
    with target.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return target


def run_with_upload(message: dict, upload, store_dir: Path) -> dict:
    """HTTP mode gets its input as the `content` upload; it is saved to the store and removed after."""
    saved = save_upload(upload, store_dir) if upload is not None and MODE == "http" else None
    try:
        return run_task(message, storage.storage_for(message, saved, store_dir))
    finally:
        if saved is not None:
            saved.unlink(missing_ok=True)


async def process(message: dict, form, store_dir: Path):
    upload = form.get("content") if hasattr(form, "get") else None
    if upload is not None and not hasattr(upload, "file"):
        upload = None
    # Model fitting blocks; keep the event loop free for /health.
    return await asyncio.to_thread(run_with_upload, message, upload, store_dir)


app = create_md_service_app(title="MD Gensim Service", process_handler=process, store_dir_env="GENSIM_STORE_DIR")


logger.info("md-gensim (%s)", "DEBUG" if DEBUG else "INFO")
_md_path = os.environ.get("MD_PATH", "").strip()
if MODE == "http":
    logger.info("storage: http (adapter elg) - inputs are uploaded, outputs served from /files; set MD_PATH for disk mode")
elif not _md_path:
    logger.warning("STORAGE_MODE=disk but MD_PATH is not set: every task will fail.")
elif not (Path(_md_path) / "data").is_dir():
    logger.warning("MD_PATH=%s has no data/ directory: input files will not be found.", _md_path)
else:
    logger.info("storage: disk (adapter elg_fs) under %s", _md_path)


@app.get("/help")
async def help_markdown() -> PlainTextResponse:
    help_path = HERE / "index.md"
    if not help_path.exists():
        return PlainTextResponse("# MD Gensim\n\nHelp file is missing.\n", status_code=404, media_type="text/markdown")
    return PlainTextResponse(help_path.read_text(encoding="utf-8"), media_type="text/markdown")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9009")))
