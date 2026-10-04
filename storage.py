"""Where inputs come from and outputs go (same rules as MD-embeddings and MD-bertopic).

disk  inputs are read under MD_PATH, outputs written to <MD_PATH>/data/<db>/tmp and returned by
      file name (`elg_fs` adapter);
http  the input arrives as the `content` upload (a set as a zip of its files by label); outputs stay
      in the store and are returned as /files URIs (`elg` adapter). Used when the service has no
      access to MessyDesk's disk.
"""

import os
import zipfile
from collections.abc import Iterator
from pathlib import Path

from fastapi import HTTPException


def storage_mode() -> str:
    """STORAGE_MODE=disk|http; by default disk when MD_PATH is set, http otherwise."""
    explicit = os.environ.get("STORAGE_MODE", os.environ.get("FILE_STORAGE_MODE", "")).strip().lower()
    if explicit in ("disk", "http"):
        return explicit
    return "disk" if os.environ.get("MD_PATH", "").strip() else "http"


def adapter_for(mode: str) -> str:
    return "elg_fs" if mode == "disk" else "elg"


def md_root() -> Path:
    raw = os.environ.get("MD_PATH", "").strip()
    if not raw:
        raise HTTPException(status_code=500, detail="MD_PATH is not set")
    return Path(raw).resolve()


def resolve_input(relative: str) -> Path:
    if not relative or not str(relative).strip():
        raise HTTPException(status_code=400, detail="Missing file path")
    if os.path.isabs(relative):
        raise HTTPException(status_code=400, detail="File path must be relative to MD_PATH")
    root = md_root()
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=400, detail="File path is outside MD_PATH")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Input file not found: {relative}")
    return resolved


def tmp_dir_for(relative: str) -> Path:
    """data/<db>/tmp of the database the input file belongs to (where the backend looks)."""
    parts = Path(relative).parts
    for i, part in enumerate(parts[:-1]):
        if part == "data":
            target = md_root() / "data" / parts[i + 1] / "tmp"
            target.mkdir(parents=True, exist_ok=True)
            return target
    raise HTTPException(status_code=400, detail=f"Cannot find data/<db> in path: {relative}")


TEXT_EXTENSIONS = {"txt", "md", "text"}


def decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def is_text(entry: dict) -> bool:
    """Sets can hold images and other files too; only their texts are read."""
    if entry.get("type") == "text":
        return True
    extension = str(entry.get("extension") or str(entry.get("label") or "").rsplit(".", 1)[-1]).lower()
    return extension in TEXT_EXTENSIONS


# A text of the input: its file entry ({@rid, label, ...}) and its text (None when unreadable).
Text = tuple[dict, str | None]


class Disk:
    def __init__(self, message: dict):
        self.message = message

    def input(self) -> Path:
        return resolve_input((self.message.get("file") or {}).get("path"))

    def texts(self) -> Iterator[Text]:
        """The texts of a whole-set job (`files`), or the job's one file."""
        entries = self.message.get("files")
        if not entries:
            file = self.message.get("file") or {}
            yield file, decode(self.input().read_bytes())
            return
        for entry in entries:
            if not is_text(entry):
                continue
            try:
                yield entry, decode(resolve_input(entry.get("path")).read_bytes())
            except HTTPException:
                yield entry, None

    def output(self, name: str) -> Path:
        files = self.message.get("files") or []
        ref = (self.message.get("file") or {}).get("path") or (files[0].get("path") if files else None)
        return tmp_dir_for(ref) / name

    def respond(self, task: str, outputs: list[dict]) -> dict:
        return {"task": task, "response": {"type": "disk", "files": [
            {"path": o["name"], "label": o["label"], "type": o["type"], "extension": o["extension"]} for o in outputs
        ]}}


class Http:
    def __init__(self, message: dict, upload: Path | None, store_dir: Path):
        self.message = message
        self.upload = upload
        self.store_dir = store_dir

    def input(self) -> Path:
        if self.upload is None:
            raise HTTPException(status_code=400, detail="No input file uploaded (field 'content')")
        return self.upload

    def texts(self) -> Iterator[Text]:
        """A set arrives as a zip of its files by label; one file arrives as itself."""
        entries = self.message.get("files")
        if not entries:
            yield self.message.get("file") or {}, decode(self.input().read_bytes())
            return
        by_label = {e.get("label"): e for e in entries if e.get("label")}
        try:
            archive = zipfile.ZipFile(self.input())
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="Expected the set as a zip file") from exc
        with archive:
            for info in archive.infolist():
                if info.is_dir() or info.filename.startswith("sources/"):
                    continue
                label = Path(info.filename).name
                entry = by_label.get(label) or {"label": label}
                if is_text(entry):
                    yield entry, decode(archive.read(info))

    def output(self, name: str) -> Path:
        return self.store_dir / name

    def respond(self, task: str, outputs: list[dict]) -> dict:
        # The adapter appends ".<extension>" to the label and infers the type from the URL.
        items = []
        for o in outputs:
            label, ext = o["label"], o["extension"]
            bare = label[: -len(ext) - 1] if label.endswith("." + ext) else label
            items.append({"uri": f"/files/{o['name']}", "label": bare, "type": o["type"]})
        return {"task": task, "response": {"type": "stored", "uri": items}}


def storage_for(message: dict, upload: Path | None = None, store_dir: Path | None = None):
    if storage_mode() == "disk":
        return Disk(message)
    return Http(message, upload, store_dir or Path("./store"))
