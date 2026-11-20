from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np


SESSION_SCHEMA = 1


def _pack(value, arrays: dict[str, np.ndarray]):
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            return {"__list__": [_pack(item, arrays) for item in value.tolist()]}
        name = f"array_{len(arrays):04d}"
        arrays[name] = value
        return {"__array__": name}
    if isinstance(value, np.generic):
        return _pack(value.item(), arrays)
    if isinstance(value, dict):
        return {str(key): _pack(item, arrays) for key, item in value.items()}
    if isinstance(value, list) and len(value) >= 256 and all(
        isinstance(item, (int, float, np.integer, np.floating)) for item in value
    ):
        name = f"array_{len(arrays):04d}"
        arrays[name] = np.asarray(value)
        return {"__list_array__": name}
    if isinstance(value, (list, tuple)):
        return [_pack(item, arrays) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot store session value of type {type(value).__name__}")


def _unpack(value, arrays: dict[str, np.ndarray]):
    if isinstance(value, list):
        return [_unpack(item, arrays) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__array__"}:
            name = value["__array__"]
            if name not in arrays:
                raise ValueError(f"Session is missing array {name}")
            return arrays[name]
        if set(value) == {"__list_array__"}:
            name = value["__list_array__"]
            if name not in arrays:
                raise ValueError(f"Session is missing array {name}")
            return arrays[name].tolist()
        if set(value) == {"__list__"}:
            return [_unpack(item, arrays) for item in value["__list__"]]
        return {key: _unpack(item, arrays) for key, item in value.items()}
    return value


def save_session(path: str | Path, snapshot: dict) -> None:
    arrays: dict[str, np.ndarray] = {}
    payload = {
        "schema_version": SESSION_SCHEMA,
        "filename": str(snapshot.get("filename", "light_curve.txt")),
        "fields": _pack(snapshot.get("fields", {}), arrays),
        "result": _pack(snapshot.get("result"), arrays),
        "evidence": _pack(snapshot.get("evidence", {}), arrays),
    }
    metadata = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=True).encode("utf-8")
    array_buffer = io.BytesIO()
    np.savez_compressed(array_buffer, **arrays)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session.json", metadata)
        archive.writestr("analysis_arrays.npz", array_buffer.getvalue())
        archive.writestr("light_curve.bin", snapshot["file_bytes"])


def load_session(path: str | Path) -> dict:
    with zipfile.ZipFile(path, "r") as archive:
        required = {"session.json", "analysis_arrays.npz", "light_curve.bin"}
        if not required.issubset(set(archive.namelist())):
            raise ValueError("This is not a Periodicity Workbench session file.")
        if any(item.file_size > 700_000_000 for item in archive.infolist()):
            raise ValueError("Session file exceeds the supported size limit.")
        payload = json.loads(archive.read("session.json").decode("utf-8"))
        if payload.get("schema_version") != SESSION_SCHEMA:
            raise ValueError("This Periodicity Workbench session version is not supported.")
        file_bytes = archive.read("light_curve.bin")
        with np.load(io.BytesIO(archive.read("analysis_arrays.npz")), allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
    return {
        "filename": payload.get("filename", "light_curve.txt"),
        "file_bytes": file_bytes,
        "fields": _unpack(payload.get("fields", {}), arrays),
        "result": _unpack(payload.get("result"), arrays),
        "evidence": _unpack(payload.get("evidence", {}), arrays),
    }
