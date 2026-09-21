"""Download three CDC pages and preserve their source information."""

import hashlib
import json
import ssl
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi

PROJECT = Path(__file__).parent

SOURCES = [
    {
        "id": "cdc_diabetes_basics",
        "title": "Diabetes Basics",
        "url": "https://www.cdc.gov/diabetes/about/index.html",
    },
    {
        "id": "cdc_type_2_diabetes",
        "title": "Type 2 Diabetes",
        "url": (
            "https://www.cdc.gov/diabetes/about/"
            "about-type-2-diabetes.html"
        ),
    },
    {
        "id": "cdc_diabetes_symptoms",
        "title": "Symptoms of Diabetes",
        "url": "https://www.cdc.gov/diabetes/signs-symptoms/index.html",
    },
]


def download_page(url):
    """Fetch a page and return its bytes and response metadata."""
    request = Request(
        url,
        headers={
            "User-Agent": "HealthcareRAG-LearningProject/0.1",
            "Accept": "text/html",
        },
    )

    # Use certifi's trusted roots while keeping HTTPS verification enabled.
    context = ssl.create_default_context(cafile=certifi.where())

    with urlopen(request, timeout=60, context=context) as response:
        content_type = response.headers.get_content_type()

        if content_type != "text/html":
            raise ValueError(
                f"Expected HTML, received {content_type}"
            )

        html = response.read()

        if not html.strip():
            raise ValueError("The server returned an empty page.")

        return {
            "html": html,
            "final_url": response.geturl(),
            "encoding": response.headers.get_content_charset(),
        }


def main():
    # Each run gets its own folder, preserving earlier downloads.
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    output_folder = PROJECT / "data" / "raw" / timestamp
    output_folder.mkdir(parents=True, exist_ok=False)

    manifest = []
    successes = 0

    for source in SOURCES:
        print(f"Downloading: {source['title']}", flush=True)

        record = {
            **source,
            "publisher": "CDC",
            "downloaded_at": None,
            "source_page_date": None,
            "status": "pending",
        }

        try:
            page = download_page(source["url"])
            filename = f"{source['id']}.html"

            (output_folder / filename).write_bytes(page["html"])

            record.update({
                "status": "downloaded",
                "downloaded_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "filename": filename,
                "final_url": page["final_url"],
                "encoding": page["encoding"],
                "bytes": len(page["html"]),
                "sha256": hashlib.sha256(
                    page["html"]
                ).hexdigest(),
            })

            successes += 1
            print(f"  Saved {len(page['html']):,} bytes.")

        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            record.update({
                "status": "failed",
                "error": str(error),
            })
            print(f"  Failed: {error}")

        manifest.append(record)

        # Preserve progress after each source.
        (output_folder / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nDownloaded {successes}/{len(SOURCES)} pages.")
    print(f"Output folder: {output_folder}")
    print("Source information saved in manifest.json.")

    if successes != len(SOURCES):
        raise SystemExit(
            "Some downloads failed. Check manifest.json for details."
        )


if __name__ == "__main__":
    main()
