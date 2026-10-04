import ast
import asyncio
import contextlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse


ProcessHandler = Callable[[dict, Any, Path], Union[Awaitable[dict], dict]]

logger = logging.getLogger("md-service-boilerplate")

# Stored files aren't deleted on fetch (consumer retries could hit a 404), so a sweep is the
# only cleanup - anything older than this is assumed abandoned (fetched already or never claimed).
STORE_MAX_AGE_SECONDS = int(os.environ.get("STORE_MAX_AGE_SECONDS", str(60 * 60)))
STORE_CLEANUP_INTERVAL_SECONDS = int(os.environ.get("STORE_CLEANUP_INTERVAL_SECONDS", str(5 * 60)))


def get_service_port() -> int:
    return int(os.environ.get("PORT", "9010"))


def _ensure_store_dir(store_dir: Path) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def _cleanup_stale_store(store_dir: Path, max_age_seconds: int) -> None:
    cutoff = time.time() - max_age_seconds
    try:
        entries = list(store_dir.iterdir())
    except FileNotFoundError:
        return
    for entry in entries:
        if entry.name.startswith("."):
            continue  # e.g. the .gitignore that keeps the store directory in git
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            logger.info(json.dumps({"event": "store_cleanup_removed", "path": str(entry)}))
        except FileNotFoundError:
            continue
        except Exception as err:
            logger.warning(json.dumps({"event": "store_cleanup_failed", "path": str(entry), "error": str(err)}))


async def _store_cleanup_loop(store_dir: Path, max_age_seconds: int, interval_seconds: int) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        _cleanup_stale_store(store_dir, max_age_seconds)


def _load_descriptor(
    descriptor_path: Path,
    service_id: Optional[str],
    service_name: Optional[str],
    service_adapter: Optional[str],
    service_local_url: Optional[str],
) -> dict:
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Descriptor file not found: {descriptor_path}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Descriptor file is not valid JSON: {exc}") from exc

    if not isinstance(descriptor, dict):
        raise HTTPException(status_code=500, detail="Descriptor root must be an object")

    if not isinstance(descriptor.get("id"), str) or not descriptor.get("id", "").strip():
        raise HTTPException(status_code=500, detail="Descriptor id must be a non-empty string")

    if "tasks" in descriptor and not isinstance(descriptor.get("tasks"), dict):
        raise HTTPException(status_code=500, detail="Descriptor tasks must be an object")

    # Runtime overrides are optional and intended for deployment-specific details.
    descriptor["id"] = service_id or descriptor["id"]
    descriptor["name"] = service_name or descriptor.get("name")
    descriptor["adapter"] = service_adapter or descriptor.get("adapter")
    descriptor["local_url"] = service_local_url or descriptor.get("local_url")

    return descriptor


async def _parse_message_from_form(form: Any) -> dict:
    message_part = form.get("message")
    if message_part is None:
        raise HTTPException(status_code=422, detail="Missing message field")

    try:
        message_value: Any = message_part
        if hasattr(message_value, "read"):
            raw_bytes = await message_value.read()
            raw_message = raw_bytes.decode("utf-8", errors="replace")
            if hasattr(message_value, "close"):
                await message_value.close()
        elif isinstance(message_value, (bytes, bytearray)):
            raw_message = bytes(message_value).decode("utf-8", errors="replace")
        else:
            raw_message = str(message_value)

        raw_message = raw_message.strip().lstrip("\ufeff")

        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            message = ast.literal_eval(raw_message)
            if not isinstance(message, dict):
                raise ValueError("message payload is not an object")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid message payload: {exc}") from exc

    return message


async def _close_extra_uploads(form: Any) -> None:
    if not hasattr(form, "multi_items"):
        return

    for key, value in form.multi_items():
        if key == "message":
            continue
        if isinstance(value, UploadFile):
            await value.close()


def create_md_service_app(
    *,
    title: str,
    process_handler: ProcessHandler,
    store_dir_env: str = "SERVICE_STORE_DIR",
    store_dir_default: str = "./store",
) -> FastAPI:
    store_dir = _ensure_store_dir(Path(os.environ.get(store_dir_env, store_dir_default)).resolve())
    descriptor_path = Path(os.environ.get("SERVICE_DESCRIPTOR_PATH", "./service.json")).resolve()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        _cleanup_stale_store(store_dir, STORE_MAX_AGE_SECONDS)  # clear anything left over from a previous run
        task = asyncio.create_task(_store_cleanup_loop(store_dir, STORE_MAX_AGE_SECONDS, STORE_CLEANUP_INTERVAL_SECONDS))
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    app = FastAPI(title=title, lifespan=lifespan)

    service_id = os.environ.get("SERVICE_ID")
    service_name = os.environ.get("SERVICE_NAME")
    service_adapter = os.environ.get("SERVICE_ADAPTER")
    service_local_url = os.environ.get("SERVICE_LOCAL_URL", f"http://localhost:{get_service_port()}")

    @app.get("/health")
    def health() -> dict:
        descriptor = _load_descriptor(
            descriptor_path=descriptor_path,
            service_id=service_id,
            service_name=service_name,
            service_adapter=service_adapter,
            service_local_url=service_local_url,
        )
        return {"status": "ok", "service": descriptor.get("id")}

    @app.get("/config")
    def config() -> dict:
        return _load_descriptor(
            descriptor_path=descriptor_path,
            service_id=service_id,
            service_name=service_name,
            service_adapter=service_adapter,
            service_local_url=service_local_url,
        )

    @app.post("/process")
    async def process(request: Request):
        form = await request.form()
        message = await _parse_message_from_form(form)
        # Uploads stay open until the handler is done: HTTP-mode handlers read `content`.
        try:
            result = process_handler(message, form, store_dir)
            if hasattr(result, "__await__"):
                return await result
            return result
        finally:
            await _close_extra_uploads(form)

    @app.get("/files/{filename}")
    def get_file(filename: str):
        file_path = (store_dir / filename).resolve()
        if file_path.parent != store_dir or not file_path.exists():
            return JSONResponse(status_code=404, content={"error": "File not found"})
        return FileResponse(path=file_path, filename=filename, media_type="text/plain")

    return app
