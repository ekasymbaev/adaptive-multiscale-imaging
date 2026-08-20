#!/usr/bin/env python3
"""Download, verify, and selectively extract the M-A island SEM dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset_audit.json"),
        help="Dataset configuration relative to the repository root.",
    )
    return parser.parse_args()


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urllib.request.urlopen(request) as response:
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        if offset and not resumed:
            print("Server did not honor the range request; restarting the partial file.")
            offset = 0
        transferred = offset
        next_report = transferred + 64 * 1024 * 1024
        with partial.open(mode) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                transferred += len(chunk)
                if transferred >= next_report:
                    print(f"Downloaded {transferred / (1024**2):.1f} MiB", flush=True)
                    next_report += 64 * 1024 * 1024
    os.replace(partial, destination)


def safe_extract_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, root: Path) -> None:
    relative = PurePosixPath(member.filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extract(member, root)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    paths = config["paths"]

    raw_dir = project_root / paths["raw_dir"]
    extracted_dir = project_root / paths["extracted_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / source["archive_name"]

    if archive_path.exists():
        current_md5 = md5sum(archive_path)
        if current_md5 != source["archive_md5"]:
            raise ValueError(
                f"Existing archive checksum mismatch at {archive_path}; "
                "preserved the file rather than overwriting it."
            )
        print(f"Archive already present and verified: {archive_path}")
    else:
        print(f"Downloading {source['download_url']}")
        download_with_resume(source["download_url"], archive_path)
        current_md5 = md5sum(archive_path)
        if current_md5 != source["archive_md5"]:
            raise ValueError(
                f"Downloaded archive MD5 mismatch: expected {source['archive_md5']}, "
                f"found {current_md5}"
            )
        print(f"Checksum verified: {current_md5}")

    if archive_path.stat().st_size != source["archive_size_bytes"]:
        raise ValueError(
            f"Archive size mismatch: expected {source['archive_size_bytes']}, "
            f"found {archive_path.stat().st_size}"
        )

    prefix = paths["native_subset"].rstrip("/") + "/"
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Corrupt ZIP member: {corrupt_member}")
        selected = [member for member in archive.infolist() if member.filename.startswith(prefix)]
        image_members = [
            member
            for member in selected
            if not member.is_dir() and PurePosixPath(member.filename).suffix.lower() in {".tif", ".tiff"}
        ]
        label_members = [
            member
            for member in selected
            if not member.is_dir() and PurePosixPath(member.filename).suffix.lower() == ".json"
        ]
        if len(image_members) != config["expected"]["image_count"] or len(label_members) != 2:
            raise ValueError(
                f"Unexpected native subset structure: {len(image_members)} images, "
                f"{len(label_members)} JSON files"
            )
        for member in selected:
            target = extracted_dir / member.filename
            if target.exists():
                continue
            safe_extract_member(archive, member, extracted_dir)

    source_metadata = {
        "dataset_name": config["dataset_name"],
        **source,
        "downloaded_archive_md5": current_md5,
        "native_subset": paths["native_subset"],
        "native_image_count": len(image_members),
        "annotation_file_count": len(label_members),
    }
    (raw_dir / "source_metadata.json").write_text(
        json.dumps(source_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Native subset ready: {len(image_members)} images and "
        f"{len(label_members)} annotation files under {extracted_dir / prefix}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; a partial download can be resumed on the next run.", file=sys.stderr)
        raise SystemExit(130)
