"""Download every dataset and model the pipeline needs into ``data/raw/`` (S10).

Each download is recorded in ``data/raw/MANIFEST.json`` with its URL, UTC
timestamp, byte size, SHA-256 and a license note.  The script is idempotent:
a file already present and recorded is skipped unless ``--force`` is passed.
Nothing is written outside ``data/`` and the Hugging Face cache.

Groups (``--groups``, comma separated, default ``all``):

``muse``        MUSE en-hi / hi-en dictionaries (train, test, full)
``flores``      FLORES-200 dev + devtest, Hindi and English
``xquad``       XQuAD validation, Hindi and English, as JSONL
``wikipedia``   Hindi Wikipedia dump (English is streamed later, not stored)
``fasttext``    pretrained aligned fastText vectors (``ft_ref``)
``kaikki``      Kaikki Hindi Wiktionary extract
``miracl``      MIRACL-hi dev topics, qrels and corpus (optional, S33 only)
``models``      pre-fetch encoder / MT weights into the Hugging Face cache
``stanza``      Stanza Hindi and English models

Licenses are **not** guessed: each entry is either the license stated on the
source page or marked ``unverified`` for a human to check.  See
``docs/DATA.md`` for the collected notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"
RAW_ROOT = REPO_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_ROOT / "MANIFEST.json"

MUSE_BASE = "https://dl.fbaipublicfiles.com/arrival/dictionaries"
USER_AGENT = "hi-en-semantic-search/0.1 (course project; data download)"

LICENSES = {
    "muse": "unverified: MUSE dictionaries (Conneau et al. 2018); check the MUSE README before redistribution",
    "flores": "unverified: FLORES-200 (NLLB team); check the flores200_dataset repository licence",
    "xquad": "unverified: XQuAD (Artetxe et al. 2020); CC-BY-SA 4.0 stated in the deepmind/xquad repository",
    "wikipedia": "unverified: Wikipedia dump, CC-BY-SA; cite the dump date",
    "fasttext": "unverified: fastText pretrained vectors, CC-BY-SA 3.0 stated on fasttext.cc",
    "kaikki": "unverified: Kaikki/Wiktionary extract, CC-BY-SA 3.0 / GFDL; check kaikki.org",
    "miracl": "unverified: MIRACL (Zhang et al. 2023); check the miracl/miracl dataset card and the Wikipedia licence of the corpus",
    "models": "unverified: model licence of the upstream Hugging Face repository applies",
    "stanza": "unverified: Stanza models, Apache-2.0 for the code; check the model card",
}

LOGGER = logging.getLogger("download_data")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load ``configs/data.yaml``."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with seconds precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load the manifest, or return an empty one when it does not exist yet."""
    if not path.exists():
        return {"generated_at": None, "files": {}, "models": {}}
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest.setdefault("files", {})
    manifest.setdefault("models", {})
    return manifest


def save_manifest(manifest: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    """Write the manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["generated_at"] = utc_now()
    temporary = path.with_suffix(".json.part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def relative(path: Path) -> str:
    """Return ``path`` relative to the repository root, POSIX-style."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def download_file(url: str, destination: Path, *, timeout: float = 60.0) -> None:
    """Stream ``url`` to ``destination`` through a ``.part`` file.

    Raises ``RuntimeError`` on HTTP or filesystem errors; the partially written
    file is removed so an interrupted download is never mistaken for a complete
    one.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    LOGGER.info("downloading %s", url)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as out:
            expected = response.headers.get("Content-Length")
            shutil.copyfileobj(response, out, length=1 << 20)
        size = temporary.stat().st_size
        if expected is not None and int(expected) != size:
            raise RuntimeError(f"truncated download: {size} of {expected} bytes ({url})")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    LOGGER.info("saved %s (%.1f MB)", relative(destination), destination.stat().st_size / 1e6)


def fetch(
    url: str,
    destination: Path,
    group: str,
    manifest: dict[str, Any],
    *,
    force: bool = False,
    license_note: str | None = None,
) -> Path:
    """Download ``url`` unless it is already recorded, then update the manifest."""
    key = relative(destination)
    entry = manifest["files"].get(key)

    recorded = entry is not None and entry.get("url") == url
    if force or not destination.exists() or not recorded:
        if destination.exists() and not force:
            LOGGER.warning("%s exists but is not the file this manifest expects; re-downloading", key)
        download_file(url, destination)
    else:
        LOGGER.info("skip %s (already downloaded)", key)

    manifest["files"][key] = {
        "url": url,
        "group": group,
        "downloaded_at": utc_now(),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_of(destination),
        "license": license_note or LICENSES[group],
    }
    return destination


def download_muse(config: dict[str, Any], manifest: dict[str, Any], *, force: bool) -> None:
    """Fetch the MUSE dictionaries for every configured language pair."""
    directory = REPO_ROOT / config["muse"]["dir"]
    train_split = config["muse"]["splits"]["train"]
    test_split = config["muse"]["splits"]["test"]
    for pair in config["muse"]["pairs"]:
        names = [f"{pair}.txt", f"{pair}.{train_split}.txt", f"{pair}.{test_split}.txt"]
        for name in names:
            fetch(f"{MUSE_BASE}/{name}", directory / name, "muse", manifest, force=force)
    save_manifest(manifest)


def _find_member(archive: tarfile.TarFile, wanted: str) -> tarfile.TarInfo:
    """Return the archive member whose path ends with ``wanted``."""
    matches = [member for member in archive.getmembers() if member.name.endswith(wanted)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one archive member ending in {wanted!r}, found {len(matches)}")
    return matches[0]


def download_flores(config: dict[str, Any], manifest: dict[str, Any], *, force: bool) -> None:
    """Fetch the FLORES-200 tarball and extract the four Hindi/English files."""
    settings = config["flores"]
    directory = REPO_ROOT / settings["dir"]
    tarball = fetch(
        settings["tarball_url"], directory / "flores200_dataset.tar.gz", "flores", manifest, force=force
    )

    targets = []
    for split in settings["splits"]:
        for language_code in settings["languages"].values():
            targets.append((f"{split}/{language_code}.{split}", directory / split / f"{language_code}.{split}"))

    missing = [(member, path) for member, path in targets if not path.exists()]
    if not missing and not force:
        LOGGER.info("skip FLORES extraction (files already present)")
        save_manifest(manifest)
        return

    with tarfile.open(tarball, "r:gz") as archive:
        for member_name, destination in targets:
            if destination.exists() and not force:
                continue
            member = _find_member(archive, member_name)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"archive member is not a regular file: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with extracted, destination.open("wb") as out:
                shutil.copyfileobj(extracted, out)
            manifest["files"][relative(destination)] = {
                "url": f"{settings['tarball_url']}::{member.name}",
                "group": "flores",
                "downloaded_at": utc_now(),
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_of(destination),
                "license": LICENSES["flores"],
            }
            LOGGER.info("extracted %s", relative(destination))
    save_manifest(manifest)


def download_xquad(config: dict[str, Any], manifest: dict[str, Any], *, force: bool) -> None:
    """Fetch XQuAD validation data as JSONL, from Hugging Face with a GitHub fallback."""
    settings = config["xquad"]
    directory = REPO_ROOT / settings["dir"]
    for language, config_name in settings["configs"].items():
        destination = directory / f"xquad.{language}.jsonl"
        if destination.exists() and not force:
            LOGGER.info("skip %s (already downloaded)", relative(destination))
            continue
        records = _xquad_from_hub(config_name, settings["split"]) or _xquad_from_github(language)
        if not records:
            raise RuntimeError(f"could not load {config_name}: both the Hub and GitHub fallbacks failed")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest["files"][relative(destination)] = {
            "url": f"hf://datasets/{settings['hf_dataset']}/{config_name}::split={settings['split']}",
            "group": "xquad",
            "downloaded_at": utc_now(),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_of(destination),
            "license": LICENSES["xquad"],
            "n_records": len(records),
        }
        LOGGER.info("wrote %s (%d records)", relative(destination), len(records))
    save_manifest(manifest)


def _xquad_from_hub(config_name: str, split: str) -> list[dict[str, Any]]:
    """Load one XQuAD config from the Hugging Face Hub, or return an empty list."""
    try:
        from datasets import load_dataset
    except ImportError:
        LOGGER.warning("datasets is not installed; skipping the Hub path for %s", config_name)
        return []
    try:
        dataset = load_dataset("google/xquad", name=config_name, split=split)
    except Exception as error:  # noqa: BLE001 - any Hub failure falls back to GitHub
        LOGGER.warning("Hub load failed for %s (%s); trying the GitHub fallback", config_name, error)
        return []
    return [dict(record) for record in dataset]


def _xquad_from_github(language: str) -> list[dict[str, Any]]:
    """Load ``xquad.<lang>.json`` from the deepmind/xquad repository."""
    url = f"https://raw.githubusercontent.com/deepmind/xquad/master/xquad.{language}.json"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as handle:
        payload = json.load(handle)
    return [dict(record) for record in payload["data"]]


def download_wikipedia(config: dict[str, Any], manifest: dict[str, Any], *, force: bool) -> None:
    """Download the Hindi Wikipedia shard files; English is streamed, never stored.

    Only the Hindi side is materialized (``AGENTS.md`` §4, memory budget); the
    English side is streamed from the Hub inside ``preprocess_wiki.py``.  The
    dump date is the one configured in ``configs/data.yaml``.
    """
    settings = config["wikipedia"]
    directory = REPO_ROOT / settings["dir"]
    dump_date = settings.get("dump_date", "20231101")
    repository = "wikimedia/wikipedia"
    config_name = f"{dump_date}.hi"
    files = _hf_repo_files(repository, config_name)
    if not files:
        raise RuntimeError(f"could not list {repository}/{config_name} on the Hub")
    for filename in files:
        destination = directory / config_name / filename
        fetch(
            f"https://huggingface.co/datasets/{repository}/resolve/main/{config_name}/{filename}",
            destination,
            "wikipedia",
            manifest,
            force=force,
            license_note=f"{LICENSES['wikipedia']} (dump date {dump_date})",
        )

    note_path = directory / "STREAM.en.json"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note = {
        "dataset": repository,
        "config": f"{dump_date}.en",
        "split": "train",
        "mode": "streaming",
        "note": "English Wikipedia is streamed in preprocess_wiki.py and never stored (~5.4M articles, >20 GB).",
    }
    note_path.write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
    manifest["files"][relative(note_path)] = {
        "url": f"hf://datasets/{repository}/{dump_date}.en::streaming",
        "group": "wikipedia",
        "downloaded_at": utc_now(),
        "size_bytes": note_path.stat().st_size,
        "sha256": sha256_of(note_path),
        "license": LICENSES["wikipedia"],
    }
    save_manifest(manifest)


def _hf_repo_files(repository: str, path_in_repo: str) -> list[str]:
    """Return the filenames in one Hugging Face dataset directory."""
    url = f"https://huggingface.co/api/datasets/{repository}/tree/main/{path_in_repo}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as handle:
            entries = json.load(handle)
    except urllib.error.URLError as error:
        LOGGER.warning("Hub listing failed for %s/%s: %s", repository, path_in_repo, error)
        return []
    return [entry["path"].rsplit("/", 1)[-1] for entry in entries if entry.get("type") == "file"]


def download_fasttext(
    config: dict[str, Any], manifest: dict[str, Any], *, force: bool, with_cc: bool
) -> None:
    """Fetch the aligned wiki vectors (``ft_ref``) and optionally the cc vectors."""
    settings = config["fasttext_pretrained"]
    directory = REPO_ROOT / settings["dir"]
    for language, url in settings["ref"].items():
        fetch(url, directory / Path(url).name, "fasttext", manifest, force=force)
    if with_cc:
        for language, url in settings["cc"].items():
            fetch(url, directory / Path(url).name, "fasttext", manifest, force=force)
    else:
        LOGGER.info("skipping the optional Common Crawl vectors (pass --with-cc to fetch them)")
    save_manifest(manifest)


def download_kaikki(config: dict[str, Any], manifest: dict[str, Any], *, force: bool) -> None:
    """Fetch the Kaikki Hindi Wiktionary JSONL extract used by A4 (S37)."""
    settings = config["kaikki"]
    directory = REPO_ROOT / settings["dir"]
    filename = settings["hi_jsonl"]
    url = f"https://kaikki.org/dictionary/Hindi/{filename}"
    fetch(url, directory / filename, "kaikki", manifest, force=force)
    save_manifest(manifest)


def download_miracl(config: dict[str, Any], manifest: dict[str, Any], *, force: bool) -> None:
    """Fetch the MIRACL-hi dev topics, qrels and corpus shards (S33, optional).

    Only the ``hi`` dev split is fetched: 350 dev queries, their qrels (752
    relevant judgments over 734 passages) and the full Hindi corpus, which is
    about half a million short passages in two gzipped JSONL shards.
    """
    settings = config["miracl"]
    directory = REPO_ROOT / settings["dir"]
    language = settings["language"]
    split = settings["split"]
    repository = settings["hf_dataset"]
    base = f"https://huggingface.co/datasets/{repository}/resolve/main"
    corpus_repository = settings.get("corpus_hf_dataset", "miracl/miracl-corpus")

    files = {
        "topics": f"miracl-v1.0-{language}/topics/topics.miracl-v1.0-{language}-{split}.tsv",
        "qrels": f"miracl-v1.0-{language}/qrels/qrels.miracl-v1.0-{language}-{split}.tsv",
    }
    for kind, path_in_repo in files.items():
        fetch(f"{base}/{path_in_repo}", directory / f"{kind}.miracl-v1.0-{language}-{split}.tsv", "miracl", manifest, force=force)

    corpus_base = f"https://huggingface.co/datasets/{corpus_repository}/resolve/main/miracl-corpus-v1.0-{language}"
    names = _hf_repo_files(corpus_repository, f"miracl-corpus-v1.0-{language}")
    if not names:
        raise RuntimeError(f"could not list {corpus_repository}/miracl-corpus-v1.0-{language} on the Hub")
    for name in sorted(names):
        if not name.endswith(".jsonl.gz"):
            continue
        fetch(f"{corpus_base}/{name}", directory / "corpus" / name, "miracl", manifest, force=force)
    save_manifest(manifest)


def prefetch_models(manifest: dict[str, Any], *, force: bool) -> None:
    """Pre-fetch the encoder and MT weights into the Hugging Face cache.

    Uses ``snapshot_download`` so no model is loaded and no GPU memory is used;
    the cache location is recorded, not the bytes (they live outside the repo).
    """
    model_ids = [
        "sentence-transformers/LaBSE",
        "intfloat/multilingual-e5-base",
        "google/muril-base-cased",
        "Helsinki-NLP/opus-mt-hi-en",
    ]
    for model_id in model_ids:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("huggingface_hub is required to pre-fetch models") from error
        LOGGER.info("pre-fetching %s", model_id)
        path = snapshot_download(repo_id=model_id, resume_download=True, force_download=force)
        manifest["models"][model_id] = {
            "cache_path": str(path),
            "downloaded_at": utc_now(),
            "license": LICENSES["models"],
        }
    save_manifest(manifest)


def download_stanza(manifest: dict[str, Any]) -> None:
    """Download the Stanza Hindi and English pipelines."""
    try:
        import stanza
    except ImportError as error:
        raise RuntimeError("stanza is required for the Stanza models") from error
    for language in ("hi", "en"):
        LOGGER.info("downloading Stanza model for %s", language)
        stanza.download(language)
        manifest["models"][f"stanza:{language}"] = {
            "cache_path": str(Path.home() / "stanza_resources" / language),
            "downloaded_at": utc_now(),
            "license": LICENSES["stanza"],
        }
    save_manifest(manifest)


GROUP_HANDLERS: dict[str, Callable[..., None]] = {
    "muse": download_muse,
    "flores": download_flores,
    "xquad": download_xquad,
    "wikipedia": download_wikipedia,
    "fasttext": download_fasttext,
    "kaikki": download_kaikki,
    "miracl": download_miracl,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--groups",
        default="all",
        help="comma separated: all, %s, models, stanza" % ", ".join(GROUP_HANDLERS),
    )
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument(
        "--with-cc",
        action="store_true",
        help="also fetch the optional Common Crawl fastText vectors for ft_cc (about 2.3 GB more)",
    )
    parser.add_argument("--skip-models", action="store_true", help="do not pre-fetch Hugging Face weights")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def resolve_groups(specification: str, *, skip_models: bool) -> list[str]:
    """Turn the ``--groups`` string into a validated, ordered list of groups."""
    available = list(GROUP_HANDLERS) + ["models", "stanza"]
    if specification.strip() == "all":
        selected = list(available)
    else:
        selected = [part.strip() for part in specification.split(",") if part.strip()]
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise SystemExit(f"unknown groups: {unknown}; available: {available}")
    if skip_models:
        selected = [group for group in selected if group != "models"]
    ordered = [group for group in available if group in selected]
    return ordered


def main(argv: list[str] | None = None) -> int:
    """Run the selected download groups and report what landed on disk."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    config = load_config()
    groups = resolve_groups(args.groups, skip_models=args.skip_models)
    manifest = load_manifest()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    LOGGER.info("groups: %s", ", ".join(groups))
    for group in groups:
        if group == "models":
            prefetch_models(manifest, force=args.force)
        elif group == "stanza":
            download_stanza(manifest)
        elif group == "fasttext":
            download_fasttext(config, manifest, force=args.force, with_cc=args.with_cc)
        else:
            GROUP_HANDLERS[group](config, manifest, force=args.force)

    save_manifest(manifest)
    total = sum(entry["size_bytes"] for entry in manifest["files"].values())
    LOGGER.info("manifest: %d files, %.2f GB, %d models", len(manifest["files"]), total / 1e9, len(manifest["models"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
