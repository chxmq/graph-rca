#!/bin/bash
# Real Incident Data Collection - All-in-One Script
# Usage: ./collect_data.sh [setup|github|sre-weekly|stats|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# SETUP FUNCTION
# ============================================================================
setup() {
    echo -e "${BLUE}🚀 Setting up infrastructure...${NC}"
    
    # Create directories
    mkdir -p "${BASE_DIR}/sources/github_postmortems"
    mkdir -p "${BASE_DIR}/sources/sre_weekly"
    mkdir -p "${BASE_DIR}/sources/raw"
    
    # Clone GitHub post-mortems
    if [ -d "${BASE_DIR}/sources/github_postmortems/.git" ]; then
        echo -e "${YELLOW}⏭️  Repository exists, pulling updates...${NC}"
        cd "${BASE_DIR}/sources/github_postmortems" && git pull
    else
        echo -e "${GREEN}📥 Cloning GitHub post-mortems...${NC}"
        cd "${BASE_DIR}/sources"
        git clone https://github.com/danluu/post-mortems github_postmortems
    fi
    
    # Setup Python venv
    if [ ! -d "${BASE_DIR}/.venv" ]; then
        echo -e "${GREEN}🐍 Creating Python environment...${NC}"
        python3 -m venv "${BASE_DIR}/.venv"
        source "${BASE_DIR}/.venv/bin/activate"
        pip install --quiet --upgrade pip
        pip install --quiet requests beautifulsoup4
    fi
    
    echo -e "${GREEN}✅ Setup complete!${NC}"
}

# ============================================================================
# GITHUB COLLECTION FUNCTION
# ============================================================================
collect_github() {
    echo -e "${BLUE}📥 Extracting links from GitHub post-mortems README...${NC}"
    
    SOURCE_FILE="${BASE_DIR}/sources/github_postmortems/README.md"
    OUTPUT_DIR="${BASE_DIR}/sources/raw/github"
    
    if [ ! -f "${SOURCE_FILE}" ]; then
        echo -e "${RED}❌ README.md not found! Run './collect_data.sh setup' first${NC}"
        exit 1
    fi
    
    mkdir -p "${OUTPUT_DIR}"
    
    # Use Python to parse the README more reliably
    [ -d "${BASE_DIR}/.venv" ] && source "${BASE_DIR}/.venv/bin/activate"
    
    python3 << PYTHON_EOF
import re
import json
import os

readme_path = "${SOURCE_FILE}"
output_dir = "${OUTPUT_DIR}"
os.makedirs(output_dir, exist_ok=True)

with open(readme_path, 'r') as f:
    content = f.read()

# Pattern: [Company](URL). Description
# We look for lines starting with [
entries = re.findall(r'\[(.*?)\]\((.*?)\)\. (.*?)(?=\n\n|\n\[|$)', content, re.DOTALL)

incidents = []
for i, (company, url, description) in enumerate(entries):
    incident = {
        "company": company.strip(),
        "url": url.strip(),
        "description": description.strip(),
        "source": "GitHub post-mortems list"
    }
    incidents.append(incident)
    
    # Save individual metadata file for each
    with open(os.path.join(output_dir, f"incident_{i+1:03d}.json"), 'w') as f:
        json.dump(incident, f, indent=2)

print(f"  ✓ Extracted {len(incidents)} incident links from README")
PYTHON_EOF
    
    echo -e "${GREEN}✅ Extraction complete!${NC}"
}

# ============================================================================
# SRE WEEKLY COLLECTION FUNCTION
# ============================================================================
collect_sre_weekly() {
    echo -e "${BLUE}📥 Collecting from SRE Weekly...${NC}"
    
    OUTPUT_DIR="${BASE_DIR}/sources/raw/sre_weekly"
    mkdir -p "${OUTPUT_DIR}"
    
    # Activate venv
    [ -d "${BASE_DIR}/.venv" ] && source "${BASE_DIR}/.venv/bin/activate"
    
    # Create and run Python scraper using absolute paths
    python3 << PYTHON_EOF
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import time
import os

incidents = []
base_url = "https://sreweekly.com"
output_path = "${OUTPUT_DIR}/incidents.json"

print("🔍 Scraping SRE Weekly...")
for page in range(1, 11):
    url = f"{base_url}/page/{page}/" if page > 1 else base_url
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('article')
        
        page_count = 0
        for article in articles:
            text = article.get_text().lower()
            if any(kw in text for kw in ['incident', 'outage', 'postmortem', 'root cause']):
                title_elem = article.find(['h1', 'h2', 'h3'])
                link_elem = article.find('a')
                if title_elem and link_elem:
                    incidents.append({
                        'title': title_elem.get_text().strip(),
                        'url': link_elem.get('href'),
                        'source': f'SRE Weekly Page {page}'
                    })
                    page_count += 1
        print(f"  ✓ Page {page}: found {page_count} items")
        time.sleep(1)
    except Exception as e:
        print(f"  ✗ Page {page}: {e}")

with open(output_path, 'w') as f:
    json.dump(incidents, f, indent=2)

print(f"✅ Found {len(incidents)} total incidents")
PYTHON_EOF
}

# ============================================================================
# STATISTICS FUNCTION
# ============================================================================
show_stats() {
    echo -e "${BLUE}📊 Data Statistics${NC}"
    echo "================================="
    
    # Raw data counts
    RAW_GITHUB=$(ls -1 "${BASE_DIR}/sources/raw/github"/*.json 2>/dev/null | wc -l | tr -d ' ')
    RAW_SRE=$([ -f "${BASE_DIR}/sources/raw/sre_weekly/incidents.json" ] && python3 -c "import json; print(len(json.load(open('${BASE_DIR}/sources/raw/sre_weekly/incidents.json'))))" || echo "0")
    
    echo -e "Raw GitHub Links Found: ${GREEN}${RAW_GITHUB}${NC}"
    echo -e "Raw SRE Weekly Items:   ${GREEN}${RAW_SRE}${NC}"
    echo ""
    
    TOTAL=$(find "${BASE_DIR}" -maxdepth 1 -type d -name "incident_*" | wc -l | tr -d ' ')
    echo -e "Organized Incidents:    ${GREEN}${TOTAL}${NC}"
    
    if [ "${TOTAL}" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  No organized incidents found yet${NC}"
        echo "Curation process:"
        echo "  1. Browse links in sources/raw/"
        echo "  2. Use collect_incidents.py to create incident_XXX folders"
        return
    fi
    
    # Python stats
    [ -d "${BASE_DIR}/.venv" ] && source "${BASE_DIR}/.venv/bin/activate"
    
    python3 << 'PYTHON_EOF'
import json
from pathlib import Path
from collections import Counter

base_dir = Path(".")
incidents = list(base_dir.glob("incident_*/"))

# File completeness
complete = sum(1 for i in incidents if all([
    (i/"logs.txt").exists(),
    (i/"postmortem.md").exists(),
    (i/"metadata.json").exists(),
    (i/"ground_truth.json").exists()
]))

print(f"\n📁 Complete incidents: {complete}/{len(incidents)}")

# Categories
categories = []
for inc in incidents:
    gt = inc / "ground_truth.json"
    if gt.exists():
        with open(gt) as f:
            categories.append(json.load(f).get('category', 'Unknown'))

if categories:
    print("\n📂 Categories:")
    for cat, count in Counter(categories).most_common():
        print(f"  {cat}: {count}")
PYTHON_EOF
}

# ============================================================================
# MAIN
# ============================================================================
case "${1:-help}" in
    setup)
        setup
        ;;
    github)
        collect_github
        ;;
    sre-weekly)
        collect_sre_weekly
        ;;
    stats)
        show_stats
        ;;
    all)
        setup
        collect_github
        collect_sre_weekly
        show_stats
        ;;
    *)
        echo "Usage: $0 [setup|github|sre-weekly|stats|all]"
        echo ""
        echo "Commands:"
        echo "  setup       - Initialize environment and clone repos"
        echo "  github      - Collect from GitHub post-mortems"
        echo "  sre-weekly  - Collect from SRE Weekly"
        echo "  stats       - Show collection statistics"
        echo "  all         - Run everything"
        echo ""
        echo "Quick start: $0 all"
        exit 1
        ;;
esac
