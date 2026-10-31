"""Strict UTF-8 JSON reading and writing.

Reading rejects: invalid UTF-8, a BOM, duplicate object keys, ``NaN``/``Infinity``
and numbers that overflow to infinity. Writing keeps Arabic readable
(``ensure_ascii=False``) and never overwrites an existing file unless asked.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class JsonFormatError(ValueError):
    pass


def _reject_constant(name: str) -> Any:
    raise JsonFormatError(f"non-finite number {name} is not allowed")


def _parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise JsonFormatError(f"number {text} is not finite")
    return value


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JsonFormatError(f"duplicate key {key!r}")
        result[key] = value
    return result


def loads_strict(data: str | bytes) -> Any:
    if isinstance(data, bytes):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JsonFormatError(f"invalid UTF-8: {exc}") from exc
    if not isinstance(data, str):
        raise JsonFormatError(f"expected str or bytes, got {type(data).__name__}")
    try:
        return json.loads(
            data,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
            object_pairs_hook=_no_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise JsonFormatError(str(exc)) from exc


def read_json_file(path: str | Path) -> Any:
    return loads_strict(Path(path).read_bytes())


def dumps(obj: Any, *, indent: int | None = 2) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent, allow_nan=False)


def write_json_file(path: str | Path, obj: Any, *, overwrite: bool = False) -> None:
    """Write ``obj`` as UTF-8 JSON. Raises ``FileExistsError`` rather than overwrite."""
    mode = "w" if overwrite else "x"
    with open(path, mode, encoding="utf-8", newline="\n") as fh:
        fh.write(dumps(obj))
        fh.write("\n")
