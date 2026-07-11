"""Download PlantVillage and PlantDoc, then extract them into ordinary files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from plant_disease.paths import DOWNLOADS_DIR, RAW_DIR, ensure_project_directories

DATASETS = {
    "plantvillage": "mohanty/PlantVillage",
    "plantdoc": "agyaatcoder/PlantDoc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--overwrite-extracted", action="store_true")
    return parser.parse_args()


def download_snapshot(name: str, force_download: bool) -> Path:
    destination = DOWNLOADS_DIR / name
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASETS[name]} to {destination}")
    snapshot_download(
        repo_id=DATASETS[name],
        repo_type="dataset",
        local_dir=destination,
        force_download=force_download,
    )
    return destination


def extract_plantvillage(snapshot: Path, overwrite: bool) -> None:
    destination = RAW_DIR / "PlantVillage"
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        print(f"PlantVillage already exists at {destination}; skipping extraction")
        return
    if overwrite and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    archives = sorted(snapshot.rglob("*.zip"))
    if archives:
        for archive in archives:
            print(f"Extracting {archive.name}")
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(destination)
    else:
        image_files = [
            path
            for path in snapshot.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not image_files:
            files = HfApi().list_repo_files(DATASETS["plantvillage"], repo_type="dataset")
            raise RuntimeError(
                "No ZIP archive or images were found in the downloaded PlantVillage snapshot. "
                f"Repository contains {len(files)} files; inspect {snapshot}."
            )
        for source in image_files:
            relative = source.relative_to(snapshot)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f"PlantVillage extracted to {destination}")


def extract_plantdoc(snapshot: Path, overwrite: bool) -> None:
    parquet_files = sorted(snapshot.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No PlantDoc Parquet shards found in {snapshot}")
    parquet_dir = DOWNLOADS_DIR / "PlantDoc-parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for source in parquet_files:
        target = parquet_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)

    command = [
        sys.executable,
        str(Path(__file__).with_name("extract_plantdoc.py")),
        "--parquet-dir",
        str(parquet_dir),
        "--output-dir",
        str(RAW_DIR / "PlantDoc"),
    ]
    if overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    ensure_project_directories()
    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name in selected:
        snapshot = download_snapshot(name, args.force_download)
        if name == "plantvillage":
            extract_plantvillage(snapshot, args.overwrite_extracted)
        else:
            extract_plantdoc(snapshot, args.overwrite_extracted)


if __name__ == "__main__":
    main()
