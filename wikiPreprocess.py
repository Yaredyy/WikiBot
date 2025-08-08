import mwxml
import re
from pathlib import Path

# Input & output paths
DUMP_FILE = "enwiki-latest-pages-articles.xml"  # your Wikipedia XML
OUTPUT_FILE = "wiki_clean.txt"

def clean_text(text):
    """Remove markup, extra whitespace, and non-ASCII characters."""
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)  # Remove templates
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"\[\[File:.*?\]\]", "", text)  # Remove file links
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]", r"\1", text)  # Wiki links
    text = re.sub(r"={2,}.*?={2,}", "", text)  # Section headers
    text = re.sub(r"\n+", "\n", text)  # Collapse newlines
    text = re.sub(r"[^ -~\n]", "", text)  # Remove non-ASCII
    return text.strip()

def preprocess_wiki():
    print("Starting preprocessing...")
    output_path = Path(OUTPUT_FILE)
    with output_path.open("w", encoding="utf-8") as out_f:
        dump = mwxml.Dump.from_file(open(DUMP_FILE, "rb"))
        for page in dump:
            for revision in page:
                if revision.text:
                    cleaned = clean_text(revision.text)
                    if cleaned:
                        out_f.write(cleaned + "\n")
    print(f"Done! Saved clean text to {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_wiki()
