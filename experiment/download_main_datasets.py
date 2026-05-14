#!/usr/bin/env python3
"""
Download and prepare two main RAG document datasets:

1. Python 3.12 Chinese official documentation, HTML archive.
2. MDN Web Docs English content, Markdown archive.

Recommended location:
    experiment/download_main_datasets.py

Run from project root:
    python experiment/download_main_datasets.py --force

Outputs:
    experiment/data/clusters/python_zh_312_html/*.html
    experiment/data/clusters/mdn_web_docs_en_md/*.md
    experiment/data/dataset_manifest.json

Design notes:
- Output files are flattened into each cluster root because the current
  experiment uploader scans only direct child files of each cluster directory.
- The script preserves source path and title information in a manifest.
- Only Python standard library modules are used.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw_sources"
CLUSTER_DIR = DATA_DIR / "clusters"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"

PYTHON_DOWNLOAD_PAGE = "https://docs.python.org/zh-cn/3.12/download.html"
MDN_ZIP_URL = "https://github.com/mdn/content/archive/refs/heads/main.zip"

PYTHON_CLUSTER = "python_zh_312_html"
MDN_CLUSTER = "mdn_web_docs_en_md"

DEFAULT_TARGET_COUNT = 200
DEFAULT_MIN_SELECTED = 100
DEFAULT_MAX_SELECTED = 300
DEFAULT_MIN_CHARS = 1200
DEFAULT_MAX_CHARS = 80000


@dataclass(frozen=True)
class CandidateDoc:
    source_path: str
    title: str
    text_chars: int
    group: str
    score: float
    suffix: str


@dataclass
class DatasetStats:
    dataset: str
    source_url: str
    archive_path: str
    raw_file_count: int
    candidate_count: int
    selected_count: int
    output_dir: str
    output_suffix: str
    min_chars: int
    max_chars: int
    avg_chars: float
    groups: dict[str, int]
    sample_files: list[str]
    warnings: list[str]


def log(message: str) -> None:
    print(f"[dataset] {message}", flush=True)


def fail(message: str) -> None:
    raise RuntimeError(message)


def ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            fail(f"Directory already exists: {path}. Use --force to overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def http_get_bytes(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RAG-thesis-dataset-downloader/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def discover_python_html_zip_url() -> str:
    log(f"Discovering Python HTML archive from {PYTHON_DOWNLOAD_PAGE}")
    page_bytes = http_get_bytes(PYTHON_DOWNLOAD_PAGE)
    page_text = page_bytes.decode("utf-8", errors="ignore")

    match = re.search(r'href="([^"]*python-3\.12[^"]*docs-html\.zip)"', page_text)
    if not match:
        match = re.search(r'href="([^"]*docs-html\.zip)"', page_text)
    if not match:
        fail("Could not discover Python HTML zip URL from the download page.")

    return urllib.parse.urljoin(PYTHON_DOWNLOAD_PAGE, html.unescape(match.group(1)))


def download_file(url: str, target_path: Path, force: bool, min_bytes: int) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and not force:
        if target_path.stat().st_size >= min_bytes:
            log(f"Archive already exists and looks large enough: {target_path}")
            return
        fail(f"Existing file is too small: {target_path}. Use --force to re-download.")

    log(f"Downloading: {url}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "RAG-thesis-dataset-downloader/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response, open(
            target_path, "wb"
        ) as file:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            last_report = time.time()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report > 2:
                    if total:
                        log(
                            f"  downloaded {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MiB"
                        )
                    else:
                        log(f"  downloaded {downloaded / 1024 / 1024:.1f} MiB")
                    last_report = now
    except urllib.error.URLError as error:
        fail(f"Download failed: {url}\n{error}")

    size = target_path.stat().st_size
    if size < min_bytes:
        fail(f"Downloaded file is too small: {target_path}, size={size} bytes")
    log(f"Downloaded archive: {target_path} ({size / 1024 / 1024:.1f} MiB)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_zip(path: Path) -> None:
    log(f"Validating zip integrity: {path}")
    if not zipfile.is_zipfile(path):
        fail(f"Not a valid zip file: {path}")
    with zipfile.ZipFile(path) as zip_ref:
        bad_file = zip_ref.testzip()
        if bad_file is not None:
            fail(f"Zip corruption detected in {path}: {bad_file}")


def extract_zip(path: Path, extract_dir: Path) -> None:
    ensure_clean_dir(extract_dir, force=True)
    log(f"Extracting: {path} -> {extract_dir}")
    with zipfile.ZipFile(path) as zip_ref:
        zip_ref.extractall(extract_dir)


def strip_html_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_markdown_text(text: str) -> str:
    text = re.sub(r"(?s)^---\s*.*?\s*---", " ", text, count=1)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[#>*_{}|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_html_title(raw: str, fallback: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    if match:
        title = strip_html_text(match.group(1))
        title = re.sub(r"\s+—\s+Python.*$", "", title)
        title = re.sub(r"\s+-\s+Python.*$", "", title)
        if title:
            return title[:120]
    h1 = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", raw)
    if h1:
        title = strip_html_text(h1.group(1))
        if title:
            return title[:120]
    return Path(fallback).stem[:120]


def extract_markdown_front_matter(raw: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    match = re.match(r"(?s)^---\s*(.*?)\s*---", raw)
    if not match:
        return metadata

    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key and value:
            metadata[key] = value
    return metadata


def extract_markdown_title(raw: str, fallback: str) -> str:
    front_matter = extract_markdown_front_matter(raw)
    if front_matter.get("title"):
        return front_matter["title"][:120]
    for line in raw.splitlines():
        if line.startswith("# "):
            return line.lstrip("#").strip()[:120]
    return Path(fallback).parent.name.replace("_", " ")[:120]


def safe_flat_name(prefix: str, source_path: str, suffix: str, index: int) -> str:
    normalized = source_path.replace("\\", "/").lower()
    stem = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", normalized)
    stem = stem.strip("_")
    if len(stem) > 90:
        stem = stem[-90:]
    short_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{index:04d}_{stem}_{short_hash}{suffix}"


def group_counts(candidates: Iterable[CandidateDoc]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for doc in candidates:
        counts[doc.group] = counts.get(doc.group, 0) + 1
    return dict(sorted(counts.items()))


def select_balanced(
    candidates: list[CandidateDoc], target_count: int, group_caps: dict[str, int]
) -> list[CandidateDoc]:
    candidates = sorted(candidates, key=lambda item: (-item.score, item.source_path))
    selected: list[CandidateDoc] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}

    for doc in candidates:
        if len(selected) >= target_count:
            break
        cap = group_caps.get(doc.group, target_count)
        if counts.get(doc.group, 0) >= cap:
            continue
        selected.append(doc)
        seen.add(doc.source_path)
        counts[doc.group] = counts.get(doc.group, 0) + 1

    if len(selected) < target_count:
        for doc in candidates:
            if len(selected) >= target_count:
                break
            if doc.source_path in seen:
                continue
            selected.append(doc)
            seen.add(doc.source_path)

    return selected[:target_count]


def find_python_root(extract_dir: Path) -> Path:
    html_files = list(extract_dir.rglob("*.html"))
    if not html_files:
        fail(f"No HTML files found after Python extraction: {extract_dir}")

    candidates = []
    for path in html_files:
        if path.name == "index.html":
            root = path.parent
            if (root / "library").exists() or (root / "tutorial").exists():
                candidates.append(root)

    if candidates:
        return sorted(candidates, key=lambda p: len(str(p)))[0]

    return Path(str(Path.commonpath([str(p.parent) for p in html_files])))


def python_group(relative_path: str) -> str | None:
    first = relative_path.split("/", 1)[0]
    allowed = {
        "tutorial",
        "library",
        "reference",
        "using",
        "howto",
        "installing",
        "distributing",
        "extending",
        "faq",
    }
    return first if first in allowed else None


def is_bad_python_page(relative_path: str) -> bool:
    path = relative_path.lower().replace("\\", "/")
    name = Path(path).name
    if "/_static/" in path or "/_sources/" in path:
        return True
    if name in {
        "search.html",
        "genindex.html",
        "py-modindex.html",
        "contents.html",
        "download.html",
        "bugs.html",
        "copyright.html",
        "license.html",
        "about.html",
    }:
        return True
    if name.startswith("genindex-"):
        return True
    return False


def score_python_doc(
    group: str, relative_path: str, text_chars: int, title: str
) -> float:
    group_weight = {
        "tutorial": 110,
        "howto": 105,
        "using": 100,
        "installing": 95,
        "distributing": 90,
        "reference": 85,
        "faq": 80,
        "extending": 75,
        "library": 70,
    }.get(group, 50)
    length_score = min(text_chars / 2500, 8.0)
    if text_chars > 25000:
        length_score -= min((text_chars - 25000) / 10000, 3.0)
    bonus = 0
    lower = relative_path.lower()
    if any(
        word in lower
        for word in ["tutorial", "howto", "intro", "usage", "setup", "venv", "asyncio"]
    ):
        bonus += 4
    if Path(relative_path).name == "index.html":
        bonus += 2
    if title and len(title) >= 4:
        bonus += 1
    return group_weight + length_score + bonus


def collect_python_candidates(
    python_root: Path, min_chars: int, max_chars: int
) -> tuple[list[CandidateDoc], int]:
    candidates: list[CandidateDoc] = []
    all_html = sorted(python_root.rglob("*.html"))
    for path in all_html:
        relative_path = path.relative_to(python_root).as_posix()
        group = python_group(relative_path)
        if group is None or is_bad_python_page(relative_path):
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text_chars = len(strip_html_text(raw))
        if text_chars < min_chars or text_chars > max_chars:
            continue
        title = extract_html_title(raw, relative_path)
        candidates.append(
            CandidateDoc(
                relative_path,
                title,
                text_chars,
                group,
                score_python_doc(group, relative_path, text_chars, title),
                ".html",
            )
        )
    return candidates, len(all_html)


def copy_selected_docs(
    source_root: Path, selected: list[CandidateDoc], output_dir: Path, prefix: str
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, doc in enumerate(selected, start=1):
        source_path = source_root / doc.source_path
        target_name = safe_flat_name(prefix, doc.source_path, doc.suffix, index)
        target_path = output_dir / target_name
        shutil.copy2(source_path, target_path)
        records.append(
            {
                "filename": target_name,
                "source_path": doc.source_path,
                "title": doc.title,
                "group": doc.group,
                "text_chars": doc.text_chars,
                "sha256": sha256_file(target_path),
            }
        )
    return records


def find_mdn_root(extract_dir: Path) -> Path:
    candidates = list(extract_dir.glob("content-*/files/en-us"))
    if candidates:
        return candidates[0]
    candidates = list(extract_dir.rglob("files/en-us"))
    if candidates:
        return candidates[0]
    fail(f"Cannot find MDN files/en-us directory under {extract_dir}")


def mdn_group(relative_path: str) -> str | None:
    path = relative_path.lower().replace("\\", "/")
    if path.startswith("learn/"):
        parts = path.split("/")
        return f"learn_{parts[1]}" if len(parts) >= 2 else "learn"
    if path.startswith("web/javascript/guide/"):
        return "javascript_guide"
    if path.startswith("web/javascript/reference/"):
        return "javascript_reference"
    if path.startswith("web/html/"):
        return "html"
    if path.startswith("web/css/"):
        return "css"
    if path.startswith("web/http/"):
        return "http"
    if path.startswith("web/api/"):
        return "web_api"
    return None


def is_bad_mdn_page(relative_path: str, raw: str) -> bool:
    path = relative_path.lower().replace("\\", "/")
    name = Path(path).name
    if name != "index.md":
        return True
    if any(
        part in path
        for part in [
            "glossary/",
            "mdn/",
            "orphaned/",
            "conflicting/",
            "mozilla/",
            "games/",
            "_wikihistory",
        ]
    ):
        return True
    front_matter = extract_markdown_front_matter(raw)
    page_type = front_matter.get("page-type", "").lower()
    title = front_matter.get("title", "").lower()
    if "browser compatibility" in title:
        return True
    if page_type in {
        "css-property",
        "css-at-rule-descriptor",
        "html-attribute",
        "svg-attribute",
        "api-event",
        "web-api-event",
    }:
        return True
    if path.startswith("web/api/") and page_type not in {
        "guide",
        "landing-page",
        "overview",
    }:
        return True
    return False


def score_mdn_doc(group: str, relative_path: str, text_chars: int, title: str) -> float:
    group_weight = {
        "learn_javascript": 115,
        "learn_web": 112,
        "learn_html": 110,
        "learn_css": 110,
        "javascript_guide": 105,
        "http": 95,
        "html": 90,
        "css": 90,
        "javascript_reference": 80,
        "web_api": 70,
    }.get(group, 75)
    length_score = min(text_chars / 2500, 8.0)
    if text_chars > 30000:
        length_score -= min((text_chars - 30000) / 12000, 3.0)
    path = relative_path.lower()
    bonus = 0
    if any(
        word in path
        for word in [
            "guide",
            "using",
            "concept",
            "overview",
            "introduction",
            "tutorial",
            "how_to",
        ]
    ):
        bonus += 5
    if "reference" in path:
        bonus -= 2
    if title and len(title) >= 4:
        bonus += 1
    return group_weight + length_score + bonus


def collect_mdn_candidates(
    mdn_root: Path, min_chars: int, max_chars: int
) -> tuple[list[CandidateDoc], int]:
    candidates: list[CandidateDoc] = []
    all_md = sorted(mdn_root.rglob("index.md"))
    for path in all_md:
        relative_path = path.relative_to(mdn_root).as_posix()
        raw = path.read_text(encoding="utf-8", errors="ignore")
        group = mdn_group(relative_path)
        if group is None or is_bad_mdn_page(relative_path, raw):
            continue
        text_chars = len(strip_markdown_text(raw))
        if text_chars < min_chars or text_chars > max_chars:
            continue
        title = extract_markdown_title(raw, relative_path)
        candidates.append(
            CandidateDoc(
                relative_path,
                title,
                text_chars,
                group,
                score_mdn_doc(group, relative_path, text_chars, title),
                ".md",
            )
        )
    return candidates, len(all_md)


def build_stats(
    dataset: str,
    source_url: str,
    archive_path: Path,
    raw_file_count: int,
    candidates: list[CandidateDoc],
    selected: list[CandidateDoc],
    output_dir: Path,
    suffix: str,
    min_required: int,
    max_allowed: int,
) -> DatasetStats:
    warnings: list[str] = []
    selected_count = len(selected)
    if selected_count < min_required:
        warnings.append(
            f"selected_count={selected_count} is below min_required={min_required}"
        )
    if selected_count > max_allowed:
        warnings.append(
            f"selected_count={selected_count} is above max_allowed={max_allowed}"
        )
    if not candidates:
        warnings.append("No valid candidates were found.")
    chars = [doc.text_chars for doc in selected]
    output_files = sorted(p.name for p in output_dir.glob(f"*{suffix}"))
    if len(output_files) != selected_count:
        warnings.append(
            f"output file count mismatch: files={len(output_files)}, selected={selected_count}"
        )
    if len(output_files) != len(set(output_files)):
        warnings.append("Duplicate output filenames detected.")
    return DatasetStats(
        dataset=dataset,
        source_url=source_url,
        archive_path=str(archive_path),
        raw_file_count=raw_file_count,
        candidate_count=len(candidates),
        selected_count=selected_count,
        output_dir=str(output_dir),
        output_suffix=suffix,
        min_chars=min(chars) if chars else 0,
        max_chars=max(chars) if chars else 0,
        avg_chars=round(sum(chars) / len(chars), 2) if chars else 0.0,
        groups=group_counts(selected),
        sample_files=output_files[:5],
        warnings=warnings,
    )


def prepare_python_dataset(
    args: argparse.Namespace,
) -> tuple[DatasetStats, list[dict[str, object]]]:
    python_url = discover_python_html_zip_url()
    archive_path = RAW_DIR / "python_zh_312_docs_html.zip"
    extract_dir = RAW_DIR / "python_zh_312_html_extracted"
    output_dir = CLUSTER_DIR / PYTHON_CLUSTER
    download_file(
        python_url, archive_path, force=args.force_download, min_bytes=5 * 1024 * 1024
    )
    validate_zip(archive_path)
    extract_zip(archive_path, extract_dir)
    python_root = find_python_root(extract_dir)
    candidates, raw_count = collect_python_candidates(
        python_root, args.min_chars, args.max_chars
    )
    selected = select_balanced(
        candidates,
        args.python_count,
        {
            "tutorial": 40,
            "howto": 45,
            "using": 25,
            "installing": 15,
            "distributing": 15,
            "reference": 45,
            "faq": 25,
            "extending": 20,
            "library": 90,
        },
    )
    ensure_clean_dir(output_dir, force=True)
    records = copy_selected_docs(python_root, selected, output_dir, "python")
    stats = build_stats(
        PYTHON_CLUSTER,
        python_url,
        archive_path,
        raw_count,
        candidates,
        selected,
        output_dir,
        ".html",
        args.min_selected,
        args.max_selected,
    )
    return stats, records


def prepare_mdn_dataset(
    args: argparse.Namespace,
) -> tuple[DatasetStats, list[dict[str, object]]]:
    archive_path = RAW_DIR / "mdn_content_main.zip"
    extract_dir = RAW_DIR / "mdn_content_main_extracted"
    output_dir = CLUSTER_DIR / MDN_CLUSTER
    download_file(
        MDN_ZIP_URL, archive_path, force=args.force_download, min_bytes=50 * 1024 * 1024
    )
    validate_zip(archive_path)
    extract_zip(archive_path, extract_dir)
    mdn_root = find_mdn_root(extract_dir)
    candidates, raw_count = collect_mdn_candidates(
        mdn_root, args.min_chars, args.max_chars
    )
    selected = select_balanced(
        candidates,
        args.mdn_count,
        {
            "learn_javascript": 55,
            "learn_web": 45,
            "learn_html": 35,
            "learn_css": 35,
            "javascript_guide": 45,
            "javascript_reference": 35,
            "html": 45,
            "css": 45,
            "http": 35,
            "web_api": 25,
        },
    )
    ensure_clean_dir(output_dir, force=True)
    records = copy_selected_docs(mdn_root, selected, output_dir, "mdn")
    stats = build_stats(
        MDN_CLUSTER,
        MDN_ZIP_URL,
        archive_path,
        raw_count,
        candidates,
        selected,
        output_dir,
        ".md",
        args.min_selected,
        args.max_selected,
    )
    return stats, records


def write_manifest(
    python_stats: DatasetStats,
    python_records: list[dict[str, object]],
    mdn_stats: DatasetStats,
    mdn_records: list[dict[str, object]],
) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {
            python_stats.dataset: {
                "stats": asdict(python_stats),
                "files": python_records,
            },
            mdn_stats.dataset: {"stats": asdict(mdn_stats), "files": mdn_records},
        },
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    log(f"Wrote manifest: {MANIFEST_PATH}")


def print_stats(stats: DatasetStats) -> None:
    print()
    print(f"=== {stats.dataset} ===")
    print(f"source_url: {stats.source_url}")
    print(f"raw_file_count: {stats.raw_file_count}")
    print(f"candidate_count: {stats.candidate_count}")
    print(f"selected_count: {stats.selected_count}")
    print(f"output_dir: {stats.output_dir}")
    print(f"char_range: {stats.min_chars} - {stats.max_chars}; avg={stats.avg_chars}")
    print(f"groups: {json.dumps(stats.groups, ensure_ascii=False)}")
    print(f"sample_files: {stats.sample_files}")
    if stats.warnings:
        print("warnings:")
        for warning in stats.warnings:
            print(f"  - {warning}")
    else:
        print("warnings: none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare Python 3.12 Chinese HTML docs and MDN English Markdown docs."
    )
    parser.add_argument(
        "--python-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Number of Python HTML documents to select. Recommended: 150-250.",
    )
    parser.add_argument(
        "--mdn-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Number of MDN Markdown documents to select. Recommended: 150-250.",
    )
    parser.add_argument(
        "--min-selected",
        type=int,
        default=DEFAULT_MIN_SELECTED,
        help="Minimum acceptable selected documents per dataset.",
    )
    parser.add_argument(
        "--max-selected",
        type=int,
        default=DEFAULT_MAX_SELECTED,
        help="Maximum acceptable selected documents per dataset.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Minimum plain-text character length for a candidate document.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum plain-text character length for a candidate document.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output cluster directories. Also implies --force-download.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download archives even if existing archive files are present.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep downloaded archives and extracted raw source directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.force:
        args.force_download = True
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    if args.python_count < 1 or args.mdn_count < 1:
        fail("Document counts must be positive.")
    if args.python_count > args.max_selected or args.mdn_count > args.max_selected:
        fail("Requested document count exceeds --max-selected.")
    if args.min_chars >= args.max_chars:
        fail("--min-chars must be smaller than --max-chars.")

    python_stats, python_records = prepare_python_dataset(args)
    mdn_stats, mdn_records = prepare_mdn_dataset(args)
    write_manifest(python_stats, python_records, mdn_stats, mdn_records)
    print_stats(python_stats)
    print_stats(mdn_stats)

    all_warnings = python_stats.warnings + mdn_stats.warnings
    if all_warnings:
        print("\nCompleted with warnings. Please inspect dataset_manifest.json.")
    else:
        print("\nCompleted successfully. Both datasets passed sanity checks.")

    if not args.keep_raw:
        for extracted_dir in [
            RAW_DIR / "python_zh_312_html_extracted",
            RAW_DIR / "mdn_content_main_extracted",
        ]:
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)
        log(
            "Removed extracted raw folders. Archives are kept in experiment/data/raw_sources/."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
