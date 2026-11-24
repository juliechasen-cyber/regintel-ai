#!/usr/bin/env python3
"""Capture Streamlit UI screenshots for the README, using a headless browser.

Reproducible and scriptable, so the images in the README can be regenerated
after a UI change instead of being stale hand-taken screen grabs.

Requires a running app:
    streamlit run app/streamlit_app.py --server.port 8795
    python scripts/capture_screenshots.py

Writes PNGs to docs/images/.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "images")

QUESTION = "Who is treated as a beneficial owner of a legal entity customer?"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8795")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args(argv)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:  pip install playwright && "
              "python -m playwright install chromium")
        return 1

    os.makedirs(args.out, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=2,  # retina-quality PNGs
        )
        page.goto(args.url, wait_until="networkidle", timeout=60_000)
        # Streamlit renders over a websocket after first paint.
        page.wait_for_selector("text=Ask a regulatory question", timeout=60_000)
        page.wait_for_timeout(2500)

        shots: list[tuple[str, str]] = []

        # 1. Research tab, empty state (shows sidebar system status)
        path = os.path.join(args.out, "01-research.png")
        page.screenshot(path=path, full_page=False)
        shots.append(("01-research.png", "Research tab"))

        # 2. Retrieved passages for a real question
        page.fill("textarea", QUESTION)
        page.wait_for_timeout(500)
        page.click("text=Retrieve passages only")
        page.wait_for_selector("text=passage(s)", timeout=60_000)
        page.wait_for_timeout(2500)
        path = os.path.join(args.out, "02-retrieval.png")
        page.screenshot(path=path, full_page=True)
        shots.append(("02-retrieval.png", "Retrieved passages with scores and pages"))

        # 3. Document library
        page.click("text=Document library")
        page.wait_for_timeout(2500)
        path = os.path.join(args.out, "03-library.png")
        page.screenshot(path=path, full_page=True)
        shots.append(("03-library.png", "Document library"))

        # 4. Upload tab
        page.click("text=Upload")
        page.wait_for_timeout(1800)
        path = os.path.join(args.out, "04-upload.png")
        page.screenshot(path=path, full_page=False)
        shots.append(("04-upload.png", "Upload with verified metadata"))

        browser.close()

    print(f"wrote {len(shots)} screenshots to {args.out}")
    for name, desc in shots:
        size = os.path.getsize(os.path.join(args.out, name))
        print(f"  {name:22} {size/1024:7.0f} KB  {desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
