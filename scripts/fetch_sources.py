#!/usr/bin/env python3
"""Download the source PDFs listed in ``data/sources.json``.

For each entry this writes two files into ``data/pdfs/``:

* ``<filename>`` — the PDF itself;
* ``<filename>.meta.json`` — the verified metadata sidecar that ``ingestion.py``
  trusts for title / regulator / publication date / document type / source URL.

The sidecar exists so the pipeline never has to guess a document's provenance
from its text. Metadata comes only from this manifest; nothing is inferred.

Usage:
    python scripts/fetch_sources.py              # skip files already present
    python scripts/fetch_sources.py --force      # re-download everything
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "sources.json")
OUT_DIR = os.path.join(ROOT, "data", "pdfs")

# Generic client name on purpose: no personal or project-owner details are sent.
USER_AGENT = "RegIntelAI-research/1.0 (+https://github.com/)"

SIDECAR_FIELDS = (
    "title",
    "regulator",
    "document_type",
    "document_type_label",
    "publication_date",
    "source_url",
    "source_note",
    "version_note",
)


class BotChallenge(RuntimeError):
    """The host served a bot-protection challenge instead of the document.

    Some regulator sites (e.g. ffiec.gov, behind Cloudflare) refuse automated
    requests. We surface that plainly rather than trying to work around it: the
    PDF must be downloaded in a normal browser and dropped into data/pdfs/.
    """


def _ssl_context() -> ssl.SSLContext:
    """Build an SSL context with a usable CA bundle.

    Some Python installs (notably python.org builds on macOS) ship without
    system root certificates, which makes every HTTPS fetch fail with
    CERTIFICATE_VERIFY_FAILED. Prefer certifi's bundle when it is installed.
    """
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout,
                                    context=_ssl_context()) as response:
            content = response.read()
            mitigated = response.headers.get("cf-mitigated")
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429) and exc.headers.get("cf-mitigated"):
            raise BotChallenge(
                f"HTTP {exc.code}: host served a bot challenge — download this "
                f"one manually in a browser"
            ) from exc
        raise

    if mitigated:
        raise BotChallenge(
            "host served a bot challenge — download this one manually in a browser"
        )
    return content


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as fh:
        entries = json.load(fh)

    os.makedirs(args.out, exist_ok=True)

    ok = skipped = 0
    failures: list[tuple[str, str]] = []

    for entry in entries:
        filename = entry["filename"]
        target = os.path.join(args.out, filename)
        sidecar_path = target + ".meta.json"

        # Always (re)write the sidecar — it is cheap and keeps metadata in sync.
        sidecar = {k: entry[k] for k in SIDECAR_FIELDS if k in entry}
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=2, ensure_ascii=False)

        if os.path.exists(target) and not args.force:
            print(f"  skip  {filename} (already downloaded)")
            skipped += 1
            continue

        print(f"  get   {filename}")
        try:
            content = fetch(entry["source_url"])
            if not content.startswith(b"%PDF"):
                raise ValueError(
                    f"response is not a PDF (starts with {content[:16]!r})"
                )
            with open(target, "wb") as fh:
                fh.write(content)
            print(f"        saved {len(content):,} bytes")
            ok += 1
        except (BotChallenge, urllib.error.URLError, ValueError, OSError,
                TimeoutError) as exc:
            print(f"        FAILED: {exc}")
            failures.append((filename, str(exc)))

    print()
    print(f"{ok} downloaded, {skipped} already present, {len(failures)} failed "
          f"(of {len(entries)} in manifest)")
    if failures:
        print("\nFailures — download these in a browser and save them into "
              f"{args.out} using the exact filename shown:")
        by_url = {e["filename"]: e["source_url"] for e in entries}
        for name, err in failures:
            print(f" - {name}\n     {by_url.get(name, '')}\n     ({err})")
        print("\nThe .meta.json sidecars are already written, so once the PDF is "
              "in place its citations will carry the correct title, regulator, "
              "date and source URL.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
