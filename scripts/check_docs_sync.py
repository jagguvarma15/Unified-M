"""Minimal docs sync checks for architecture drift."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "README.md",
    ROOT / "docs" / "ARCHITECTURE.md",
]

FORBIDDEN_TERMS = {
    "Streamlit UI": "UI is React + Vite now.",
    "Streamlit-based interactive dashboard": "UI is React + Vite now.",
    "`src/ingestion/`": "Use current paths under src/connectors and src/pipeline.",
    "`src/schemas/`": "Schemas live in src/core/contracts.py and server schemas.",
    "`src/mmm/`": "Models live under src/models/.",
    "`src/api/`": "API lives under src/server/.",
    "ui/app.py": "UI source lives under ui/src/.",
}

REQUIRED_TERMS = {
    ROOT / "README.md": [
        "React",
        "FastAPI",
    ],
    ROOT / "docs" / "ARCHITECTURE.md": [
        "src/server/",
        "ui/src",
        "src/models/",
        "src/pipeline/",
    ],
}


def main() -> int:
    failures: list[str] = []
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        for term, hint in FORBIDDEN_TERMS.items():
            if term in text:
                failures.append(f"{path.relative_to(ROOT)} contains forbidden term '{term}'. {hint}")

    for path, required in REQUIRED_TERMS.items():
        text = path.read_text(encoding="utf-8")
        for term in required:
            if term not in text:
                failures.append(f"{path.relative_to(ROOT)} is missing required term '{term}'.")

    if failures:
        print("Docs sync check failed:")
        for line in failures:
            print(f"- {line}")
        return 1

    print("Docs sync check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
