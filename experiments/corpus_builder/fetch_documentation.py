#!/usr/bin/env python3
"""
Fetch real-world documentation for RAG corpus expansion.
Extracts 20-30 high-quality documents from official sources.
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

OUTPUT_DIR = Path(__file__).parent / "curated_docs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Documentation sources
SOURCES = {
    "flask": [
        "https://flask.palletsprojects.com/en/3.0.x/errorhandling/",
        "https://flask.palletsprojects.com/en/3.0.x/logging/",
        "https://flask.palletsprojects.com/en/3.0.x/deploying/",
        "https://flask.palletsprojects.com/en/3.0.x/debugging/",
        "https://flask.palletsprojects.com/en/3.0.x/config/",
    ],
    "mongodb": [
        "https://www.mongodb.com/docs/manual/reference/error-codes/",
        "https://www.mongodb.com/docs/manual/tutorial/troubleshoot-replica-sets/",
        "https://www.mongodb.com/docs/manual/reference/connection-string/",
        "https://www.mongodb.com/docs/manual/administration/connection-pool-overview/",
        "https://www.mongodb.com/docs/manual/core/read-preference/",
    ],
    "linux": [
        "https://www.kernel.org/doc/html/latest/admin-guide/sysrq.html",
        "https://www.kernel.org/doc/html/latest/admin-guide/kernel-parameters.html",
    ],
}

def fetch_and_convert(url: str, filename: str) -> None:
    """Fetch URL and convert to markdown"""
    print(f"Fetching: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main content (adjust selectors based on site)
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if not main_content:
            print(f"  ⚠️  Could not find main content in {url}")
            return
        
        # Remove navigation, ads, etc.
        for unwanted in main_content.find_all(['nav', 'aside', 'footer', 'script', 'style']):
            unwanted.decompose()
        
        # Convert to markdown-ish text
        text = main_content.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n\n'.join(lines)
        
        # Save
        output_path = OUTPUT_DIR / f"{filename}.md"
        output_path.write_text(f"# {filename.replace('_', ' ').title()}\n\nSource: {url}\n\n{clean_text}")
        
        print(f"  ✓ Saved to {output_path}")
        
    except Exception as e:
        print(f"  ✗ Error fetching {url}: {e}")

def main():
    print("Fetching real-world documentation...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    doc_count = 0
    
    for category, urls in SOURCES.items():
        for i, url in enumerate(urls, 1):
            filename = f"{category}_{i:02d}"
            fetch_and_convert(url, filename)
            doc_count += 1
            time.sleep(1)  # Be nice to servers
    
    print(f"\n✓ Fetched {doc_count} documents")
    print(f"Next: Run load_corpus.py to populate ChromaDB")

if __name__ == "__main__":
    main()
