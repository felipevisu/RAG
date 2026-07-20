"""Ingest documents/*.md into the knowledge base via the running API.

Each markdown file is split into one entry per "## " section, categorized by
the document name. Re-running duplicates entries — delete old ones first.

Usage:  python ingest.py [http://localhost:8080]
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"


def post(content: str, category: str):
    req = urllib.request.Request(
        f"{BASE}/api/posts",
        data=json.dumps({"content": content, "category": category}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as res:
        return json.load(res)["id"]


for doc in sorted(Path(__file__).parent.glob("documents/*.md")):
    category = doc.stem.replace("_", " ")
    text = doc.read_text()
    # split into sections at "## " headings; keep the heading with its body
    sections = re.split(r"(?m)^(?=## )", text)
    count = 0
    for section in sections:
        section = section.strip()
        if len(section) < 40:  # skip title-only / empty fragments
            continue
        post(section, category)
        count += 1
    print(f"{doc.name}: {count} sections ingested as category '{category}'")
