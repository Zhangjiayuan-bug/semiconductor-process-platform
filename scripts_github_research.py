from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).parent
OUTPUT = ROOT / "docs" / "github_reference_research.csv"
QUERIES = [
    "semiconductor manufacturing dataset",
    "wafer map dataset",
    "statistical process control python",
    "data drift monitoring python",
]


def fetch_json(url: str) -> dict:
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "semiconductor-process-platform"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    rows = []
    for query in QUERIES:
        url = f"https://api.github.com/search/repositories?q={quote(query)}&sort=stars&order=desc&per_page=10"
        payload = fetch_json(url)
        for item in payload.get("items", []):
            license_info = item.get("license") or {}
            rows.append(
                {
                    "query": query,
                    "repository": item["full_name"],
                    "url": item["html_url"],
                    "stars": item["stargazers_count"],
                    "license": license_info.get("spdx_id") or "NOASSERTION",
                    "language": item.get("language"),
                    "description": item.get("description") or "",
                }
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates("repository").sort_values(["query", "stars"], ascending=[True, False]).to_csv(OUTPUT, index=False)
    print(f"Wrote GitHub reference metadata to {OUTPUT}")


if __name__ == "__main__":
    main()
