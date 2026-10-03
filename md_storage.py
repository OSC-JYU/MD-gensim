"""Disk or HTTP storage mode for a MessyDesk service.

Disk mode: MessyDesk's root (the directory that contains data/) is reachable through MD_PATH.
The service reads its input from message.file.path, writes its outputs to
MD_PATH/data/<db>/tmp/ and answers {"response": {"type": "disk", "files": [...]}}. The elg_fs
consumer adapter then tells MessyDesk to move the files into place.

HTTP mode: MD_PATH is unset or has no data/ directory. The service gets its input as an upload
(the multipart "content" part) and serves outputs from /files, as the elg adapter expects.

The mode is picked once at start-up, and /config reports the matching adapter (elg_fs or elg) so
the consumer registers the service with the right one. STORAGE_MODE=http forces HTTP mode.
A request that carries a "content" upload is always handled in HTTP mode.
"""
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("md-storage")


def resolve_md_root() -> Optional[Path]:
    storage_mode = (os.getenv("STORAGE_MODE") or os.getenv("FILE_STORAGE_MODE") or "").strip().lower()
    if storage_mode == "http":
        return None
    raw = os.getenv("MD_PATH", "").strip()
    if not raw:
        return None
    root = Path(raw).resolve()
    if root.name == "data":
        root = root.parent
    if (root / "data").is_dir():
        return root
    logger.warning("MD_PATH=%s has no data/ directory, using HTTP mode", raw)
    return None


MD_ROOT = resolve_md_root()
DISK_MODE = MD_ROOT is not None


def storage_adapter() -> str:
    """The consumer adapter that matches the storage mode, reported by /config."""
    return "elg_fs" if DISK_MODE else "elg"


def describe_mode() -> str:
    return f"disk (MD_PATH={MD_ROOT})" if DISK_MODE else "http"


class StorageError(ValueError):
    """The request can't be handled in the current storage mode, or a path is not allowed."""


def resolve_input_path(relative_path: Any) -> Path:
    """Resolve a MessyDesk file path (data/<db>/...) under MD_ROOT, refusing anything outside it."""
    if not DISK_MODE:
        raise StorageError("No file uploaded as 'content', and disk mode is off (MD_PATH not found)")
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise StorageError("message.file.path is missing")
    if os.path.isabs(relative_path):
        raise StorageError("message.file.path must be relative to MD_PATH")
    data_dir = (MD_ROOT / "data").resolve()
    resolved = (MD_ROOT / relative_path).resolve()
    if data_dir not in resolved.parents:
        raise StorageError("message.file.path is outside MD_PATH/data")
    if not resolved.is_file():
        raise StorageError(f"File not found: {relative_path}")
    return resolved


def message_input_path(message: Dict[str, Any], key: Optional[str] = None) -> Path:
    """Input file of the message: message.file.path, or message.file[key].path (e.g. key="source")."""
    file_info = message.get("file") if isinstance(message.get("file"), dict) else {}
    if key:
        file_info = file_info.get(key) if isinstance(file_info.get(key), dict) else {}
    return resolve_input_path(file_info.get("path"))


def tmp_dir(message: Dict[str, Any]) -> Path:
    """MD_ROOT/data/<db>/tmp, with <db> taken from message.file.path as MessyDesk does."""
    source_path = str((message.get("file") or {}).get("path") or "")
    parts = [p for p in source_path.replace("\\", "/").split("/") if p]
    db_name = "messydesk"
    for i, part in enumerate(parts[:-1]):
        if part == "data" and parts[i + 1]:
            db_name = parts[i + 1]
            break
    directory = MD_ROOT / "data" / db_name / "tmp"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def source_label(message: Dict[str, Any], fallback: str = "file") -> str:
    label = (message.get("file") or {}).get("label")
    return label if isinstance(label, str) and label else fallback


def stage_output(message: Dict[str, Any], path: Path, label: str, file_type: str, extension: str,
                 **extra: Any) -> Dict[str, Any]:
    """Move an output file into MessyDesk's tmp/ and describe it for the disk response.

    The tmp name gets a unique prefix so parallel jobs can't overwrite each other; MessyDesk
    uses the label, not the tmp name, as the file's name.
    """
    target = tmp_dir(message) / f"{uuid.uuid4().hex}_{Path(path).name}"
    shutil.move(str(path), str(target))
    entry = {"path": target.name, "label": label, "type": file_type, "extension": extension}
    entry.update(extra)
    return entry


def write_output(message: Dict[str, Any], data: bytes, filename: str, label: str, file_type: str,
                 extension: str, **extra: Any) -> Dict[str, Any]:
    """Write output bytes straight into MessyDesk's tmp/ and describe them for the disk response."""
    target = tmp_dir(message) / f"{uuid.uuid4().hex}_{filename}"
    target.write_bytes(data)
    entry = {"path": target.name, "label": label, "type": file_type, "extension": extension}
    entry.update(extra)
    return entry


def disk_response(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"response": {"type": "disk", "files": files}}
