"""Extract article sections from our three saved CDC pages."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup


PROJECT = Path(__file__).parent
INPUT_FOLDER = PROJECT / "data" / "raw" / "manual_cdc"
OUTPUT_FOLDER = PROJECT / "data" / "processed"

SOURCES = [
    {
        "id": "cdc_diabetes_basics",
        "filename": "Diabetes Basics _ Diabetes _ CDC.html",
        "url": "https://www.cdc.gov/diabetes/about/index.html",
    },
    {
        "id": "cdc_type_2_diabetes",
        "filename": "Type 2 Diabetes _ Diabetes _ CDC.html",
        "url": (
            "https://www.cdc.gov/diabetes/about/"
            "about-type-2-diabetes.html"
        ),
    },
    {
        "id": "cdc_diabetes_symptoms",
        "filename": "Symptoms of Diabetes _ Diabetes _ CDC.html",
        "url": "https://www.cdc.gov/diabetes/signs-symptoms/index.html",
    },
]


def clean_text(element):
    """Collapse extra whitespace while keeping readable text."""
    return " ".join(element.get_text(" ", strip=True).split())


def extract_document(source):
    path = INPUT_FOLDER / source["filename"]

    if not path.exists():
        raise ValueError(f"Missing file: {path}")

    raw_html = path.read_bytes()
    soup = BeautifulSoup(raw_html, "html.parser")

    title_element = soup.select_one(".cdc-page-title h1")
    date_element = soup.select_one(
        ".cdc-page-title-bar time[datetime]"
    )

    article_sections = soup.select(
        ".cdc-dfe-body__top .dfe-section, "
        ".cdc-dfe-body__center .dfe-section"
    )

    if title_element is None or not article_sections:
        raise ValueError(
            f"Could not identify the article in {path.name}. "
            "Check that you saved the article page."
        )

    sections = []

    for section in article_sections:
        heading = section.find("h2")
        heading_text = clean_text(heading) if heading else "Overview"

        # Exclude the resources section containing video and signup links.
        if heading_text.lower() == "resources":
            continue

        # Remove navigation, related-reading links, and hidden content.
        unwanted = section.select(
            ".dfe-section__nav, .dfe-block--keep_reading, "
            "script, style, noscript, iframe, "
            ".visually-hidden, [hidden]"
        )

        for element in unwanted:
            element.extract()

        blocks = []

        for element in section.select(
            "h2, h3, h4, p, li, .cdc-callout__content"
        ):
            # A parent block already includes its children's text.
            # Skip nested blocks to avoid duplicating that text.
            parents = element.find_parents()

            if any(
                parent.name in {"p", "li"}
                or "cdc-callout__content"
                in parent.get("class", [])
                for parent in parents
            ):
                continue

            text = clean_text(element)

            if text:
                if element.name == "li":
                    text = "- " + text
                blocks.append(text)

        if blocks:
            sections.append({
                "heading": heading_text,
                "text": "\n\n".join(blocks),
            })

    if not sections:
        raise ValueError(f"No article text extracted from {path.name}")

    document = {
        "id": source["id"],
        "title": clean_text(title_element),
        "publisher": "CDC",
        "source_url": source["url"],
        "source_file": str(path.relative_to(PROJECT)),
        "source_sha256": hashlib.sha256(raw_html).hexdigest(),
        "source_reviewed_date": (
            date_element["datetime"].split()[0]
            if date_element else None
        ),
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
    }

    return document


def main():
    # Validate all three documents before writing the combined output.
    documents = [
        extract_document(source)
        for source in SOURCES
    ]

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_FOLDER / "cdc_documents.json"
    output_file.write_text(
        json.dumps(documents, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for document in documents:
        text = "\n\n".join(
            section["text"]
            for section in document["sections"]
        )

        # A text version makes manual inspection easier.
        preview_file = OUTPUT_FOLDER / f"{document['id']}.txt"
        preview_file.write_text(
            f"{document['title']}\n"
            f"Source: {document['source_url']}\n\n"
            f"{text}\n",
            encoding="utf-8",
        )

        print(
            f"{document['title']}: "
            f"{len(document['sections'])} sections, "
            f"approximately {len(text.split())} words"
        )

    print(f"\nSaved structured documents: {output_file}")
    print("Open the .txt files to check the extracted content.")


if __name__ == "__main__":
    main()