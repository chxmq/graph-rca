#!/usr/bin/env python3
"""
Real Incidents Dataset Manager

Utility for managing the real-world incident dataset used in GraphRCA experiments.

Commands:
  python manage.py stats           Show dataset statistics
  python manage.py validate        Validate all incident data files
  python manage.py generate-logs   Generate synthetic logs for incidents missing them
  python manage.py export          Export dataset summary to JSON
  python manage.py collect         Collect incident links from GitHub/SRE Weekly
  python manage.py setup           Setup environment and clone repos
"""

import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

# Fix SSL issues on some systems
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']


class IncidentDataset:
    """Manages the real-world incident dataset."""
    
    REQUIRED_FILES = ["logs.txt", "postmortem.md", "metadata.json", "ground_truth.json"]
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.sources_dir = self.base_dir / "sources"
        self.incidents = self._discover_incidents()
    
    def _discover_incidents(self) -> List[Path]:
        """Find all incident directories."""
        return sorted(self.base_dir.glob("incident_*"))
    
    # =========================================================================
    # COLLECTION & SETUP
    # =========================================================================
    
    def setup(self):
        """Initialize environment and clone repositories."""
        print("🚀 Setting up infrastructure...")
        
        # Create directories
        (self.sources_dir / "github_postmortems").mkdir(parents=True, exist_ok=True)
        (self.sources_dir / "sre_weekly").mkdir(parents=True, exist_ok=True)
        (self.sources_dir / "raw" / "github").mkdir(parents=True, exist_ok=True)
        (self.sources_dir / "raw" / "sre_weekly").mkdir(parents=True, exist_ok=True)
        
        # Clone GitHub post-mortems
        pm_dir = self.sources_dir / "github_postmortems"
        if (pm_dir / ".git").exists():
            print("⏭️  Repository exists, pulling updates...")
            subprocess.run(["git", "pull"], cwd=pm_dir, check=True)
        else:
            print("📥 Cloning GitHub post-mortems...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/danluu/post-mortems",
                str(pm_dir)
            ], check=True)
        
        print("✅ Setup complete!")
    
    def collect_github(self):
        """Extract incident links from GitHub post-mortems README."""
        import re
        
        print("📥 Extracting links from GitHub post-mortems README...")
        
        source_file = self.sources_dir / "github_postmortems" / "README.md"
        output_dir = self.sources_dir / "raw" / "github"
        
        if not source_file.exists():
            print("❌ README.md not found! Run 'python manage.py setup' first")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(source_file, 'r') as f:
            content = f.read()
        
        # Pattern: [Company](URL). Description
        entries = re.findall(r'\[(.*?)\]\((.*?)\)\. (.*?)(?=\n\n|\n\[|$)', content, re.DOTALL)
        
        for i, (company, url, description) in enumerate(entries):
            incident = {
                "company": company.strip(),
                "url": url.strip(),
                "description": description.strip()[:500],
                "source": "GitHub post-mortems list"
            }
            
            output_file = output_dir / f"incident_{i+1:03d}.json"
            with open(output_file, 'w') as f:
                json.dump(incident, f, indent=2)
        
        print(f"✓ Extracted {len(entries)} incident links from README")
        print(f"✅ Saved to {output_dir}")
    
    def collect_sre_weekly(self, pages: int = 10):
        """Scrape incident links from SRE Weekly."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            print("❌ Install dependencies: pip install requests beautifulsoup4")
            return
        
        print("📥 Collecting from SRE Weekly...")
        
        output_dir = self.sources_dir / "raw" / "sre_weekly"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        incidents = []
        base_url = "https://sreweekly.com"
        
        for page in range(1, pages + 1):
            url = f"{base_url}/page/{page}/" if page > 1 else base_url
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all('article')
                
                page_count = 0
                for article in articles:
                    text = article.get_text().lower()
                    keywords = ['incident', 'outage', 'postmortem', 'root cause', 'failure']
                    if any(kw in text for kw in keywords):
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
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"  ✗ Page {page}: {e}")
        
        output_file = output_dir / "incidents.json"
        with open(output_file, 'w') as f:
            json.dump(incidents, f, indent=2)
        
        print(f"✅ Found {len(incidents)} total incidents, saved to {output_file}")
    
    # =========================================================================
    # INCIDENT CREATION
    # =========================================================================
    
    def create_incident(
        self,
        incident_id: str,
        logs: Optional[str] = None,
        postmortem: Optional[str] = None,
        metadata: Optional[Dict] = None,
        ground_truth: Optional[Dict] = None
    ):
        """Create a complete incident entry."""
        incident_dir = self.base_dir / f"incident_{incident_id}"
        incident_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Creating incident_{incident_id}...")
        
        if logs:
            (incident_dir / "logs.txt").write_text(logs)
            print(f"  ✓ Saved logs.txt")
        
        if postmortem:
            (incident_dir / "postmortem.md").write_text(postmortem)
            print(f"  ✓ Saved postmortem.md")
        
        if metadata:
            with open(incident_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ Saved metadata.json")
        
        if ground_truth:
            # Validate required fields
            required = ["root_cause", "category"]
            for field in required:
                if field not in ground_truth:
                    raise ValueError(f"Missing required field: {field}")
            ground_truth["incident_id"] = incident_id
            
            with open(incident_dir / "ground_truth.json", 'w') as f:
                json.dump(ground_truth, f, indent=2)
            print(f"  ✓ Saved ground_truth.json")
        
        print(f"✅ Incident {incident_id} created successfully")
        return incident_dir
    
    # =========================================================================
    # STATISTICS & VALIDATION
    # =========================================================================
    
    def stats(self) -> Dict:
        """Compute dataset statistics."""
        total = len(self.incidents)
        complete = 0
        categories = []
        companies = []
        missing_files = Counter()
        
        for inc_dir in self.incidents:
            is_complete = True
            for f in self.REQUIRED_FILES:
                if not (inc_dir / f).exists():
                    missing_files[f] += 1
                    is_complete = False
            if is_complete:
                complete += 1
            
            gt_path = inc_dir / "ground_truth.json"
            if gt_path.exists():
                try:
                    with open(gt_path) as f:
                        gt = json.load(f)
                    categories.append(gt.get("category", "Unknown"))
                    companies.append(gt.get("company", "Unknown"))
                except:
                    pass
        
        return {
            "total_incidents": total,
            "complete_incidents": complete,
            "completeness_pct": round(complete / total * 100, 1) if total else 0,
            "categories": dict(Counter(categories).most_common()),
            "companies_count": len(set(companies)),
            "missing_files": dict(missing_files),
        }
    
    def validate(self) -> List[Dict]:
        """Validate all incidents and report issues."""
        issues = []
        
        for inc_dir in self.incidents:
            inc_issues = []
            inc_id = inc_dir.name
            
            for f in self.REQUIRED_FILES:
                if not (inc_dir / f).exists():
                    inc_issues.append(f"Missing {f}")
            
            gt_path = inc_dir / "ground_truth.json"
            if gt_path.exists():
                try:
                    with open(gt_path) as f:
                        gt = json.load(f)
                    required = ["root_cause", "category"]
                    for field in required:
                        if not gt.get(field):
                            inc_issues.append(f"Empty ground_truth.{field}")
                except json.JSONDecodeError:
                    inc_issues.append("Invalid ground_truth.json")
            
            logs_path = inc_dir / "logs.txt"
            if logs_path.exists():
                content = logs_path.read_text().strip()
                if len(content) < 50:
                    inc_issues.append(f"logs.txt too short ({len(content)} chars)")
            
            if inc_issues:
                issues.append({"incident": inc_id, "issues": inc_issues})
        
        return issues
    
    # =========================================================================
    # LOG GENERATION
    # =========================================================================
    
    def generate_logs(self, model: str = "llama3.2:3b"):
        """Generate synthetic logs for incidents missing them."""
        try:
            import ollama
        except ImportError:
            print("❌ Install ollama package: pip install ollama")
            return
        
        client = ollama.Client(host="http://localhost:11434")
        
        for inc_dir in self.incidents:
            logs_path = inc_dir / "logs.txt"
            
            if logs_path.exists() and logs_path.stat().st_size > 50:
                continue
            
            gt_path = inc_dir / "ground_truth.json"
            pm_path = inc_dir / "postmortem.md"
            
            if not gt_path.exists() or not pm_path.exists():
                print(f"  {inc_dir.name}: Missing ground_truth or postmortem, skipping")
                continue
            
            print(f"  {inc_dir.name}: Generating logs...")
            
            try:
                with open(gt_path) as f:
                    gt = json.load(f)
                with open(pm_path) as f:
                    postmortem = f.read()[:2000]
                
                prompt = f"""Generate realistic log entries (15-20 lines) for this incident.
Include timestamps, service names, log levels, and messages showing incident progression.

Category: {gt.get('category', 'Unknown')}
Root Cause: {gt.get('root_cause', 'Unknown')}
Context: {postmortem[:500]}

Output ONLY log lines, no markdown or explanations:"""

                response = client.generate(
                    model=model,
                    prompt=prompt,
                    options={"temperature": 0.7}
                )
                
                log_content = response["response"].strip()
                if log_content.startswith("```"):
                    log_content = "\n".join(log_content.split("\n")[1:-1])
                
                logs_path.write_text(log_content)
                print(f"  {inc_dir.name}: ✓ Saved {len(log_content)} chars")
                
            except Exception as e:
                print(f"  {inc_dir.name}: ✗ Failed: {e}")
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def export(self) -> Dict:
        """Export dataset summary."""
        incidents = []
        
        for inc_dir in self.incidents:
            inc_data = {"id": inc_dir.name}
            
            gt_path = inc_dir / "ground_truth.json"
            if gt_path.exists():
                try:
                    with open(gt_path) as f:
                        gt = json.load(f)
                    inc_data.update({
                        "category": gt.get("category"),
                        "root_cause": gt.get("root_cause"),
                        "company": gt.get("company"),
                    })
                except:
                    pass
            
            inc_data["has_logs"] = (inc_dir / "logs.txt").exists()
            inc_data["has_postmortem"] = (inc_dir / "postmortem.md").exists()
            
            incidents.append(inc_data)
        
        return {"total": len(incidents), "incidents": incidents}


def main():
    parser = argparse.ArgumentParser(
        description="Real Incidents Dataset Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage.py stats                    # Show dataset statistics
  python manage.py validate                 # Validate all incident files
  python manage.py setup                    # Clone GitHub postmortems repo
  python manage.py collect --source github  # Extract GitHub incident links
  python manage.py collect --source sre     # Scrape SRE Weekly
  python manage.py generate-logs            # Generate missing logs with LLM
  python manage.py export                   # Export summary to JSON
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Stats
    subparsers.add_parser("stats", help="Show dataset statistics")
    
    # Validate
    subparsers.add_parser("validate", help="Validate all incident files")
    
    # Setup
    subparsers.add_parser("setup", help="Setup environment and clone repos")
    
    # Collect
    collect_parser = subparsers.add_parser("collect", help="Collect incident links")
    collect_parser.add_argument("--source", choices=["github", "sre", "all"], default="all",
                               help="Source to collect from")
    collect_parser.add_argument("--pages", type=int, default=10,
                               help="Pages to scrape from SRE Weekly")
    
    # Generate logs
    gen_parser = subparsers.add_parser("generate-logs", help="Generate synthetic logs")
    gen_parser.add_argument("--model", default="llama3.2:3b", help="LLM model")
    
    # Export
    subparsers.add_parser("export", help="Export dataset summary")
    
    # Create incident
    create_parser = subparsers.add_parser("create", help="Create new incident")
    create_parser.add_argument("incident_id", help="Incident ID (e.g., 201)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    dataset = IncidentDataset()
    
    if args.command == "stats":
        stats = dataset.stats()
        print("\n" + "=" * 50)
        print("DATASET STATISTICS")
        print("=" * 50)
        print(f"Total Incidents:    {stats['total_incidents']}")
        print(f"Complete Incidents: {stats['complete_incidents']} ({stats['completeness_pct']}%)")
        print(f"Unique Companies:   {stats['companies_count']}")
        print("\nCategories:")
        for cat, count in stats["categories"].items():
            print(f"  {cat}: {count}")
        if stats["missing_files"]:
            print("\nMissing Files:")
            for f, count in stats["missing_files"].items():
                print(f"  {f}: {count} incidents")
    
    elif args.command == "validate":
        issues = dataset.validate()
        if not issues:
            print("✓ All incidents validated successfully!")
        else:
            print(f"Found issues in {len(issues)} incidents:\n")
            for item in issues[:20]:
                print(f"{item['incident']}:")
                for issue in item["issues"]:
                    print(f"  - {issue}")
    
    elif args.command == "setup":
        dataset.setup()
    
    elif args.command == "collect":
        if args.source in ["github", "all"]:
            dataset.collect_github()
        if args.source in ["sre", "all"]:
            dataset.collect_sre_weekly(pages=args.pages)
    
    elif args.command == "generate-logs":
        print(f"Generating logs with model: {args.model}")
        dataset.generate_logs(model=args.model)
        print("\n✓ Log generation complete")
    
    elif args.command == "export":
        data = dataset.export()
        output_path = dataset.base_dir / "dataset_summary.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Exported to {output_path}")
    
    elif args.command == "create":
        print(f"Creating incident_{args.incident_id}")
        print("Use the IncidentDataset.create_incident() method programmatically")
        print("or manually create the directory with required files.")


if __name__ == "__main__":
    main()
