#!/usr/bin/env python3
"""
Export FastAPI OpenAPI schema to a JSON file.

Usage:
  PYTHONPATH=src python3 scripts/export_openapi.py --out ui/openapi.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output JSON file path")
    args = parser.parse_args()

    from server.app import create_app

    app = create_app()
    schema = app.openapi()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2))
    print(f"Wrote OpenAPI schema to {out}")


if __name__ == "__main__":
    main()

