"""Download the project datasets, then extract them into ordinary files."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

from plant_disease.paths import DOWNLOADS_DIR, RAW_DIR, ensure_project_directories

DATASETS = {
    "plantvillage": "mohanty/PlantVillage",
    "plantdoc": "agyaatcoder/PlantDoc",
    "plantseg": "10.5281/zenodo.17719108",
}

PLANTSEG_URL = "https://zenodo.org/api/records/17719108/files/plantseg.zip/content"
PLANTSEG_SIZE = 1_057_281_724
PLANTSEG_MD5 = "9358a66dff88cdd15c4fe009763c40a3"
PLANTSEG_DOWNLOAD_PARTS = 8


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


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_range(
    url: str,
    destination: Path,
    start: int,
    end: int,
    progress: tqdm,
) -> None:
    expected_size = end - start + 1
    existing_size = destination.stat().st_size if destination.exists() else 0
    if existing_size > expected_size:
        destination.unlink()
        existing_size = 0
    progress.update(existing_size)
    if existing_size == expected_size:
        return

    headers = {
        "User-Agent": "plant-disease-assistant/0.1",
        "Range": f"bytes={start + existing_size}-{end}",
    }
    request = Request(url, headers=headers)
    with urlopen(request) as response, destination.open("ab") as output:
        if response.status != 206:
            raise RuntimeError(f"Zenodo did not honor range request: HTTP {response.status}")
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            progress.update(len(chunk))
    if destination.stat().st_size != expected_size:
        raise RuntimeError(
            f"Incomplete range {start}-{end}: expected {expected_size} bytes, "
            f"got {destination.stat().st_size}"
        )


def download_plantseg(force_download: bool) -> Path:
    destination = DOWNLOADS_DIR / "plantseg"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "plantseg.zip"
    if archive.exists() and not force_download:
        if archive.stat().st_size == PLANTSEG_SIZE and file_md5(archive) == PLANTSEG_MD5:
            print(f"Verified existing PlantSeg archive at {archive}")
            return destination
        print("Existing PlantSeg archive failed verification; downloading it again")

    temporary = archive.with_suffix(".zip.part")
    part_size = (PLANTSEG_SIZE + PLANTSEG_DOWNLOAD_PARTS - 1) // PLANTSEG_DOWNLOAD_PARTS
    parts = [
        temporary.with_name(f"{temporary.name}.{index:02d}")
        for index in range(PLANTSEG_DOWNLOAD_PARTS)
    ]
    if temporary.exists() and temporary.stat().st_size <= part_size and not parts[0].exists():
        temporary.replace(parts[0])
    if force_download:
        temporary.unlink(missing_ok=True)
        for part in parts:
            part.unlink(missing_ok=True)

    print(f"Downloading PlantSeg ({PLANTSEG_SIZE / 1024**3:.2f} GiB) to {archive}")
    ranges = [
        (index * part_size, min((index + 1) * part_size, PLANTSEG_SIZE) - 1)
        for index in range(PLANTSEG_DOWNLOAD_PARTS)
    ]
    with tqdm(total=PLANTSEG_SIZE, unit="B", unit_scale=True, desc="plantseg.zip") as progress:
        with ThreadPoolExecutor(max_workers=PLANTSEG_DOWNLOAD_PARTS) as executor:
            futures = [
                executor.submit(download_range, PLANTSEG_URL, part, start, end, progress)
                for part, (start, end) in zip(parts, ranges, strict=True)
            ]
            for future in futures:
                future.result()

    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)

    if temporary.stat().st_size != PLANTSEG_SIZE:
        raise RuntimeError(
            f"PlantSeg size mismatch: expected {PLANTSEG_SIZE}, got {temporary.stat().st_size}"
        )
    checksum = file_md5(temporary)
    if checksum != PLANTSEG_MD5:
        raise RuntimeError(f"PlantSeg checksum mismatch: expected {PLANTSEG_MD5}, got {checksum}")
    temporary.replace(archive)
    for part in parts:
        part.unlink(missing_ok=True)
    print(f"Verified PlantSeg checksum: {checksum}")
    return destination


def extract_zip_safely(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in tqdm(handle.infolist(), desc=f"Extracting {archive.name}"):
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Unsafe path in {archive}: {member.filename}")
            handle.extract(member, destination)


def extract_plantseg(snapshot: Path, overwrite: bool) -> None:
    destination = RAW_DIR / "PlantSeg"
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        print(f"PlantSeg already exists at {destination}; skipping extraction")
        return
    if overwrite and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = snapshot / "plantseg.zip"
    if not archive.exists():
        raise RuntimeError(f"PlantSeg archive not found at {archive}")
    extract_zip_safely(archive, destination)
    print(f"PlantSeg extracted to {destination}")


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
        snapshot = (
            download_plantseg(args.force_download)
            if name == "plantseg"
            else download_snapshot(name, args.force_download)
        )
        if name == "plantvillage":
            extract_plantvillage(snapshot, args.overwrite_extracted)
        elif name == "plantdoc":
            extract_plantdoc(snapshot, args.overwrite_extracted)
        else:
            extract_plantseg(snapshot, args.overwrite_extracted)


if __name__ == "__main__":
    main()
