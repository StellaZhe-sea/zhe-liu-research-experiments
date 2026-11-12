"""Validate that one or more generated SVG files are nonempty, parseable SVGs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


def collect(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for value in inputs:
        path = Path(value).resolve()
        if path.is_dir():
            files.extend(sorted(path.glob("*.svg")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return list(dict.fromkeys(files))


def validate(path: Path) -> None:
    if path.suffix.lower() != ".svg":
        raise ValueError(f"not an SVG file: {path}")
    if path.stat().st_size < 256:
        raise ValueError(f"SVG is unexpectedly small ({path.stat().st_size} bytes): {path}")
    root = ET.parse(path).getroot()
    if root.tag.split("}")[-1].lower() != "svg":
        raise ValueError(f"root element is not <svg>: {path}")
    if not (root.get("viewBox") or (root.get("width") and root.get("height"))):
        raise ValueError(f"SVG has neither viewBox nor width/height: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="SVG files or directories containing SVG files")
    args = parser.parse_args()
    try:
        files = collect(args.paths)
        if not files:
            raise ValueError("no SVG files found")
        for file in files:
            validate(file)
            print(f"OK  {file} ({file.stat().st_size} bytes)")
        print(f"Validated {len(files)} SVG file(s).")
        return 0
    except (OSError, ET.ParseError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
