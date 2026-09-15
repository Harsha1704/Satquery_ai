"""Admission errors and server-controlled local-file policy."""
from pathlib import Path, PureWindowsPath


class QueryError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def resolve_input(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or PureWindowsPath(value).drive or ".." in PureWindowsPath(value).parts:
        raise QueryError("invalid_input_path", "Inputs must be relative files inside the configured input root.")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise QueryError("invalid_input_path", "Input file is missing or outside the configured input root.")
    if resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".jp2"}:
        raise QueryError("invalid_input_type", "Only supported image files can be analyzed.")
    return resolved
