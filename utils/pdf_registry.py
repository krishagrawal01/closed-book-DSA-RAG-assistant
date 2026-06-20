import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data")
PDFS_DIR = DATA_DIR / "pdfs"
REGISTRY_PATH = DATA_DIR / "pdf_registry.json"
CHROMA_DIR = DATA_DIR / "chroma_db"


def ensure_data_dirs() -> None:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        with REGISTRY_PATH.open("w", encoding="utf-8") as file:
            json.dump({"pdfs": []}, file, indent=2)


def load_registry() -> list[dict]:
    ensure_data_dirs()
    with REGISTRY_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return data.get("pdfs", [])
    return data


def save_registry(entries: list[dict]) -> None:
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump({"pdfs": entries}, file, indent=2)


def make_collection_name(filename: str) -> str:
    stem = Path(filename).stem
    safe = re.sub(r"[^\w\-]", "_", stem).lower()[:40]
    suffix = hashlib.md5(filename.encode()).hexdigest()[:8]
    return f"pdf_{safe}_{suffix}"


def get_pdf_entry(filename: str) -> dict | None:
    for entry in load_registry():
        if entry["filename"] == filename:
            return entry
    return None


def add_pdf_entry(
    filename: str,
    *,
    collection_name: str,
    chunk_count: int,
    file_path: str,
) -> dict:
    entries = load_registry()
    entry = {
        "filename": filename,
        "upload_date": datetime.now(timezone.utc).isoformat(),
        "collection_name": collection_name,
        "chunk_count": chunk_count,
        "file_path": file_path,
    }
    entries.append(entry)
    save_registry(entries)
    return entry


def update_pdf_entry(filename: str, **updates) -> dict | None:
    entries = load_registry()
    for index, entry in enumerate(entries):
        if entry["filename"] == filename:
            entry.update(updates)
            entries[index] = entry
            save_registry(entries)
            return entry
    return None


def remove_pdf_entry(filename: str) -> dict | None:
    entries = load_registry()
    for index, entry in enumerate(entries):
        if entry["filename"] == filename:
            removed = entries.pop(index)
            save_registry(entries)
            return removed
    return None


def save_pdf_file(filename: str, file_bytes: bytes) -> Path:
    ensure_data_dirs()
    pdf_path = PDFS_DIR / filename
    pdf_path.write_bytes(file_bytes)
    return pdf_path


def delete_pdf_file(file_path: str) -> None:
    path = Path(file_path)
    if path.exists():
        path.unlink()
