#!/usr/bin/env python3
"""Check the paper model and mlearn snapshot bytes from the source archive."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SYSTEMS = ("Li", "Mo", "Cu", "Ni", "Si", "Ge")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    checked = 0
    for system in SYSTEMS:
        root = HERE / "lammps" / system
        manifest = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
        if manifest["system"] != system:
            raise ValueError(f"Wrong system in {root / 'model_manifest.json'}")
        for record in manifest["artifacts"]:
            path = root / record["path"]
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"Size mismatch: {path}")
            if sha256(path) != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {path}")
            checked += 1

    data_root = HERE.parents[1] / "data" / "mlearn"
    data_manifest = json.loads((data_root / "MANIFEST.json").read_text(encoding="utf-8"))
    for system in SYSTEMS:
        record = data_manifest[system]
        path = data_root / system / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"mlearn SHA-256 mismatch: {path}")

    print(f"verified {checked} promoted model artifacts and {len(SYSTEMS)} mlearn data files")


if __name__ == "__main__":
    main()
