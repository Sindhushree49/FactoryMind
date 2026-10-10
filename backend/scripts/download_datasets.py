"""
FactoryMind dataset acquisition.

Downloads the three public research datasets used by FactoryMind into
``datasets/<name>/`` and writes a provenance manifest to
``datasets/manifests/datasets.json``.

Design rules enforced here:
  * Never fabricate or synthesise data. If a download fails, the script records
    the failure in the manifest and prints manual instructions.
  * Never merge unrelated datasets. Each dataset lands in its own directory.
  * Record source URL, licence, retrieval timestamp and SHA-256 so results are
    reproducible and auditable.

Usage
-----
    python backend/scripts/download_datasets.py --dataset ai4i
    python backend/scripts/download_datasets.py --all
    python backend/scripts/download_datasets.py --all --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"
MANIFEST_PATH = DATASETS_DIR / "manifests" / "datasets.json"

USER_AGENT = "FactoryMind-DatasetFetcher/1.0 (+research; contact: repository maintainer)"

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "ai4i": {
        "title": "AI4I 2020 Predictive Maintenance Dataset",
        "publisher": "University of California, Irvine (UCI ML Repository)",
        "landing_page": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
        "doi": "10.24432/C52T6T",
        "citation": (
            "Susto, G., Paternostro, A., Bortolussi, K., De Martini, I., Nardi, F., "
            "Pagliuca, B., Pesarini, A., Pirizzini, F., Reggiani, F., and Ronci, A. (2020). "
            "Artificial Intelligence for Predictive Maintenance. Sensors, 20(21), 6164."
        ),
        "licence": "CC BY 4.0 (UCI ML Repository)",
        "download_urls": [
            "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        ],
        "archive_member": "ai4i2020.csv",
        "targets": ["ai4i/ai4i2020.csv"],
    },
    "cmapss": {
        "title": "NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set",
        "publisher": "NASA Prognostics Center of Excellence (PHM'08 competition data)",
        "landing_page": "https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data",
        "doi": None,
        "citation": (
            "Frederick, D., DeCastro, H., and Litt, J. (2007). User's Guide for the "
            "Commercial Modular Aero-Propulsion System Simulation (C-MAPSS). NASA/TM-2007-215026."
        ),
        "licence": "U.S. Government public domain (NASA)",
        "download_urls": [
            "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
        ],
        "archive_member": None,
        "members": [
            "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
            "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
            "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
            "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
        ],
        "targets": ["nasa_cmapss/cmapss/"],
    },
    "mvtec": {
        "title": "MVTec AD (Visual Anomaly Detection) Dataset",
        "publisher": "MVTec Software GmbH",
        "landing_page": "https://www.mvtec.com/company/research/datasets/mvtec-ad",
        "doi": "10.1109/CVPR.2019.00121",
        "citation": (
            "Bergmann, P., Batzfeld, K., Lange, K., Maeding, V., Missmann, S., ..., "
            "Zinkernagel, C. (2019). MVTec AD - A Comprehensive Real-World Dataset for "
            "Unsupervised Anomaly Detection. 2019 IEEE/CVF Conference on Computer Vision "
            "and Pattern Recognition (CVPR), 5392-5400."
        ),
        "licence": "CC BY-NC-SA 4.0 (research/non-commercial; registration required)",
        "download_urls": [
            # MVTec only distributes the archive after an explicit licence acceptance,
            # so no stable unauthenticated URL exists.
            "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.zip",
        ],
        "requires_manual_registration": True,
        "targets": ["mvtec_ad/mvtec_anomaly_detection/"],
    },
}


def sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"generated_at": None, "datasets": {}}


def save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _extract_zip(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)


def _flatten_nested_archives(root: Path, depth: int = 3) -> None:
    """Some publishers wrap the real payload in one or more nested archives.

    The NASA C-MAPSS S3 mirror, for example, ships
    ``<title>/CMAPSSData.zip`` inside the outer download. Unwrap repeatedly so
    callers only ever have to look for loose files.
    """
    if depth <= 0:
        return
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".zip", ".tar", ".gz", ".bz2", ".xz"}:
            destination = path.parent / f"{path.stem}_extracted"
            try:
                if path.suffix.lower() == ".zip":
                    _extract_zip(path, destination)
                elif path.suffix.lower() == ".tar":
                    with tarfile.open(path) as tf:
                        _safe_members(tf, destination)
                elif path.suffix.lower() in {".gz", ".bz2", ".xz"}:
                    mode = {".gz": "r:*", ".bz2": "r:bz2", ".xz": "r:xz"}[path.suffix.lower()]
                    destination.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(path.with_suffix(""), mode=mode) as tf:
                        _safe_members(tf, destination)
            except (zipfile.BadZipFile, tarfile.TarError, OSError):
                continue
            _flatten_nested_archives(destination, depth - 1)


def _safe_members(archive: tarfile.TarFile, dest: Path) -> None:
    """Extract while blocking path-traversal entries."""
    dest = dest.resolve()
    for member in archive.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"Unsafe tar entry: {member.name}")
    archive.extractall(dest)


def acquire_ai4i(workdir: Path) -> dict[str, Any]:
    spec = DATASET_SPECS["ai4i"]
    target = DATASETS_DIR / "ai4i" / "ai4i2020.csv"
    if target.exists() and target.stat().st_size > 100_000:
        return {"status": "cached", "path": str(target.relative_to(REPO_ROOT)), "bytes": target.stat().st_size}

    archive = workdir / "ai4i.zip"
    last_error: Exception | None = None
    for url in spec["download_urls"]:
        try:
            _download(url, archive)
            extracted = workdir / "ai4i_extract"
            if extracted.exists():
                shutil.rmtree(extracted)
            _extract_zip(archive, extracted)
            _flatten_nested_archives(extracted)
            candidates = list(extracted.rglob("*.csv"))
            if not candidates:
                raise RuntimeError("No CSV found inside archive")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidates[0], target)
            return {
                "status": "downloaded",
                "path": str(target.relative_to(REPO_ROOT)),
                "bytes": target.stat().st_size,
                "source_url": url,
                "sha256": sha256(target),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"AI4I download failed: {last_error}")


def acquire_cmapss(workdir: Path) -> dict[str, Any]:
    spec = DATASET_SPECS["cmapss"]
    target_dir = DATASETS_DIR / "nasa_cmapss" / "cmapss"
    required = [target_dir / m for m in spec["members"]]
    if all(p.exists() for p in required):
        return {
            "status": "cached",
            "path": str(target_dir.relative_to(REPO_ROOT)),
            "files": len(required),
        }

    archive = workdir / "cmapss.zip"
    last_error: Exception | None = None
    for url in spec["download_urls"]:
        try:
            _download(url, archive)
            extracted = workdir / "cmapss_extract"
            if extracted.exists():
                shutil.rmtree(extracted)
            _extract_zip(archive, extracted)
            _flatten_nested_archives(extracted)
            found = {p.name: p for p in extracted.rglob("*.txt")}
            target_dir.mkdir(parents=True, exist_ok=True)
            copied = []
            for name in spec["members"]:
                if name in found:
                    shutil.copy2(found[name], target_dir / name)
                    copied.append(name)
            if not copied:
                raise RuntimeError("No C-MAPSS .txt files found inside archive")
            return {
                "status": "downloaded",
                "path": str(target_dir.relative_to(REPO_ROOT)),
                "files": len(copied),
                "source_url": url,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"C-MAPSS download failed: {last_error}")


def acquire_mvtec(workdir: Path) -> dict[str, Any]:
    spec = DATASET_SPECS["mvtec"]
    target_dir = DATASETS_DIR / "mvtec_ad"
    if (target_dir / "mvtec_anomaly_detection").exists():
        classes = sorted(
            p.name for p in (target_dir / "mvtec_anomaly_detection").iterdir() if p.is_dir()
        )
        if len(classes) >= 10:
            return {
                "status": "cached",
                "path": str(target_dir.relative_to(REPO_ROOT)),
                "categories": classes,
            }

    archive = workdir / "mvtec_anomaly_detection.tar.xz"
    last_error: Exception | None = None
    for url in spec["download_urls"]:
        try:
            _download(url, archive)
            target_dir.mkdir(parents=True, exist_ok=True)
            if archive.suffix == ".xz" or archive.name.endswith(".tar.xz"):
                import lzma

                with lzma.open(archive) as fh:
                    with tarfile.open(fileobj=fh) as tf:
                        _safe_members(tf, target_dir)
            else:
                _extract_zip(archive, target_dir)
                _flatten_nested_archives(target_dir)
            return {
                "status": "downloaded",
                "path": str(target_dir.relative_to(REPO_ROOT)),
                "source_url": url,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(
        "MVTec AD requires manual acceptance of the non-commercial licence.\n"
        f"  Underlying error: {last_error}\n"
        "  MANUAL STEPS:\n"
        "   1. Open https://www.mvtec.com/company/research/datasets/mvtec-ad\n"
        "   2. Download 'mvtec_anomaly_detection.tar.xz' (approx. 4.9 GB unpacked)\n"
        f"   3. Extract it into {target_dir.relative_to(REPO_ROOT)}\n"
        "      so that the 15 category folders live under "
        "datasets/mvtec_ad/mvtec_anomaly_detection/\n"
        "   4. Re-run: python backend/scripts/download_datasets.py --dataset mvtec --verify"
    )


ACQUIRERS = {
    "ai4i": acquire_ai4i,
    "cmapss": acquire_cmapss,
    "mvtec": acquire_mvtec,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch FactoryMind research datasets.")
    parser.add_argument("--dataset", choices=sorted(ACQUIRERS), action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Re-hash existing files")
    args = parser.parse_args()

    selected = sorted(ACQUIRERS) if args.all else (args.dataset or [])
    if not selected:
        parser.error("provide --dataset NAME or --all")

    manifest = load_manifest()
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["generator"] = "backend/scripts/download_datasets.py"

    workdir = DATASETS_DIR / ".cache"
    workdir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in selected:
        spec = DATASET_SPECS[name]
        print(f"[FactoryMind] === {name}: {spec['title']} ===")
        entry: dict[str, Any] = {
            "title": spec["title"],
            "publisher": spec["publisher"],
            "landing_page": spec["landing_page"],
            "doi": spec.get("doi"),
            "citation": spec.get("citation"),
            "licence": spec["licence"],
            "attempts": manifest["datasets"].get(name, {}).get("attempts", []),
        }
        try:
            result = ACQUIRERS[name](workdir)
            entry.update(result)
            entry["status"] = result["status"]
            print(f"[FactoryMind] OK {name}: {result['status']} -> {result.get('path')}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            entry["status"] = "unavailable"
            entry["error"] = str(exc)
            print(f"[FactoryMind] BLOCKED {name}:\n{exc}\n")

        if args.verify and entry["status"] in {"downloaded", "cached"}:
            path = REPO_ROOT / str(entry.get("path", ""))
            if path.is_file():
                entry["sha256"] = sha256(path)
                entry["bytes"] = path.stat().st_size

        manifest["datasets"][name] = entry

    save_manifest(manifest)
    print(f"[FactoryMind] Manifest written to {MANIFEST_PATH}")
    print(f"[FactoryMind] Completed with {failures} blocker(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
