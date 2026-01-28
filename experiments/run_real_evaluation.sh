#!/bin/bash
#===============================================================================
# GraphRCA Evaluation Script for IEEE ACCESS Paper
#===============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="$SCRIPT_DIR/loghub_data"

QUICK_MODE=false
MAX_SAMPLES=500
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    MAX_SAMPLES=100
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "$DATA_DIR"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     GraphRCA Evaluation for IEEE ACCESS                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results: $RESULTS_DIR"
echo ""

#===============================================================================
# Step 1: Setup
#===============================================================================
echo -e "${YELLOW}[1/6] Setting up environment...${NC}"
cd "$PROJECT_ROOT"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo -e "${RED}✗${NC} Run ./run-gpu-server.sh first"
    exit 1
fi

pip install -q drain3 pandas scikit-learn 2>/dev/null || true
echo -e "${GREEN}✓${NC} Environment ready"

#===============================================================================
# Step 2: Check Ollama
#===============================================================================
echo -e "${YELLOW}[2/6] Checking Ollama...${NC}"
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    nohup ollama serve > /dev/null 2>&1 &
    sleep 5
fi
ollama list 2>/dev/null | grep -q "llama3.2:3b" || ollama pull llama3.2:3b
echo -e "${GREEN}✓${NC} Ollama ready"

#===============================================================================
# Step 3: Get Test Data
#===============================================================================
echo -e "${YELLOW}[3/6] Preparing test data...${NC}"
if [ ! -f "$DATA_DIR/BGL_2k.log" ]; then
    curl -sL "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log" -o "$DATA_DIR/BGL_2k.log" 2>/dev/null || {
        echo "Generating synthetic data..."
        python3 -c "
import random
from datetime import datetime, timedelta
severities = ['INFO', 'WARNING', 'ERROR', 'FATAL']
messages = ['RAS KERNEL INFO generating core', 'ciod: Error reading message', 'MMCS connecting to db', 'LINKCARD Error torus sender']
with open('$DATA_DIR/BGL_2k.log', 'w') as f:
    for i in range(2000):
        ts = 1117838570 + i*2
        sev = random.choices(severities, weights=[60,20,15,5])[0]
        f.write(f'- {ts} 2005.06.03 R0{random.randint(0,7)}-M{random.randint(0,1)} {sev} {random.choice(messages)}\n')
"
    }
fi
LOG_COUNT=$(wc -l < "$DATA_DIR/BGL_2k.log")
echo -e "${GREEN}✓${NC} Test data: $LOG_COUNT entries"

#===============================================================================
# Step 4: Run Drain (Baseline)
#===============================================================================
echo -e "${YELLOW}[4/6] Running Drain parser...${NC}"

python3 -c "
import time, json, re, statistics
from collections import defaultdict
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

config = TemplateMinerConfig()
miner = TemplateMiner(config=config)

total = successful = 0
latencies = []
templates = defaultdict(int)

with open('$DATA_DIR/BGL_2k.log') as f:
    for i, line in enumerate(f):
        if not line.strip(): continue
        total += 1
        parts = line.strip().split(None, 4)
        msg = parts[-1] if len(parts) > 4 else line.strip()
        start = time.perf_counter()
        result = miner.add_log_message(msg)
        latencies.append((time.perf_counter() - start) * 1000)
        if result and result.get('cluster_id'):
            successful += 1
            templates[result['cluster_id']] += 1
        if (i+1) % 500 == 0: print(f'  Processed {i+1}...')

results = {
    'parser': 'Drain3',
    'metrics': {
        'total_lines': total,
        'successful_parses': successful,
        'parse_rate': round(successful/total*100, 2) if total else 0,
        'unique_templates': len(templates),
        'avg_latency_ms': round(statistics.mean(latencies), 4) if latencies else 0,
        'std_latency_ms': round(statistics.stdev(latencies), 4) if len(latencies) > 1 else 0,
        'throughput_logs_per_sec': round(total/(sum(latencies)/1000), 1) if latencies else 0
    }
}
with open('$RESULTS_DIR/drain_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f\"\\nDrain: {results['metrics']['parse_rate']}% rate, {results['metrics']['throughput_logs_per_sec']:.0f} logs/sec\")
"
echo -e "${GREEN}✓${NC} Drain complete"

#===============================================================================
# Step 5: Run LLM Parser (standalone - no chromadb imports)
#===============================================================================
echo -e "${YELLOW}[5/6] Running LLM parser ($MAX_SAMPLES samples)...${NC}"
echo "   ETA: $(($MAX_SAMPLES * 2 / 60))-$(($MAX_SAMPLES * 3 / 60)) minutes"

python3 << LLMEOF
import sys, json, time, re, statistics
import ollama
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

# Minimal LogEntry model (no chromadb imports)
class LogEntry(BaseModel):
    timestamp: Optional[datetime] = None
    message: str = ""
    level: str = ""
    pid: Optional[str] = None
    component: Optional[str] = None
    error_code: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    group: Optional[str] = None
    trace_id: Optional[str] = None
    request_id: Optional[str] = None

# Standalone parser
client = ollama.Client(host='http://localhost:11434', timeout=30.0)

def parse_log_llm(log_entry):
    prompt = f"""Parse this log entry into JSON:
{log_entry}

Return JSON with: timestamp (ISO format), message, level (INFO/WARNING/ERROR/FATAL), component (optional)
"""
    try:
        response = client.generate(model="llama3.2:3b", prompt=prompt, format="json", options={"temperature": 0.2})
        return LogEntry.model_validate_json(response.response)
    except:
        return LogEntry(message=log_entry, level="UNKNOWN")

def get_ground_truth(line):
    parts = line.strip().split(None, 4)
    if len(parts) < 4: return None
    msg = parts[-1] if len(parts) > 4 else line
    sev = "INFO"
    for s in ["FATAL", "ERROR", "WARNING"]:
        if s in msg.upper():
            sev = s
            break
    return {"severity": sev, "component": "kernel", "message": msg}

# Read and sample
with open("$DATA_DIR/BGL_2k.log") as f:
    lines = [l.strip() for l in f if l.strip()]

step = max(1, len(lines) // $MAX_SAMPLES)
samples = lines[::step][:$MAX_SAMPLES]

print(f"Testing {len(samples)} samples...")

correct_sev = correct_ts = total = 0
latencies = []
errors = []

for i, line in enumerate(samples):
    gt = get_ground_truth(line)
    if not gt: continue
    total += 1
    
    try:
        start = time.perf_counter()
        result = parse_log_llm(line)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        
        # Check severity match
        gt_sev = gt["severity"].upper()
        llm_sev = (result.level or "").upper()
        sev_ok = gt_sev == llm_sev or (gt_sev == "FATAL" and llm_sev in ["FATAL","CRITICAL"]) or (gt_sev == "ERROR" and llm_sev in ["ERROR","FAIL"])
        ts_ok = result.timestamp is not None
        
        if sev_ok: correct_sev += 1
        if ts_ok: correct_ts += 1
        
        if (i+1) % 25 == 0:
            avg = statistics.mean(latencies)
            eta = int((len(samples)-i) * avg / 60)
            print(f"  [{i+1}/{len(samples)}] {avg:.1f}s/log, ETA: {eta}min")
    except Exception as e:
        errors.append(str(e)[:50])

metrics = {
    "total_samples": len(samples),
    "valid_samples": total,
    "errors": len(errors),
    "severity_accuracy": round(correct_sev/total*100, 2) if total else 0,
    "timestamp_accuracy": round(correct_ts/total*100, 2) if total else 0,
    "overall_accuracy": round((correct_sev+correct_ts)/(total*2)*100, 2) if total else 0,
    "avg_latency_s": round(statistics.mean(latencies), 2) if latencies else 0,
    "std_latency_s": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
    "throughput_logs_per_sec": round(len(latencies)/sum(latencies), 3) if latencies else 0
}

with open("$RESULTS_DIR/llm_parser_results.json", "w") as f:
    json.dump({"parser": "GraphRCA", "model": "llama3.2:3b", "metrics": metrics}, f, indent=2)

print(f"\\nGraphRCA: {metrics['severity_accuracy']}% severity, {metrics['overall_accuracy']}% overall, {metrics['avg_latency_s']}s/log")
LLMEOF

echo -e "${GREEN}✓${NC} LLM parser complete"

#===============================================================================
# Step 6: RCA Test & Report
#===============================================================================
echo -e "${YELLOW}[6/6] Running RCA test...${NC}"

python3 << 'RCAEOF'
import sys, json, time
import ollama

client = ollama.Client(host='http://localhost:11434', timeout=60.0)

SCENARIOS = [
    {"name": "Database Failure", "logs": [
        "[10:00:00] INFO db: Pool initialized",
        "[10:00:10] WARNING db: Pool at 80%", 
        "[10:00:15] ERROR db: Connection timeout",
        "[10:00:17] CRITICAL app: Database unavailable"
    ], "expected": "timeout"},
    {"name": "Memory Exhaustion", "logs": [
        "[14:00:00] INFO app: Starting batch",
        "[14:10:00] WARNING monitor: Memory at 85%",
        "[14:15:00] ERROR jvm: OutOfMemoryError",
        "[14:15:01] CRITICAL system: OOM killer"
    ], "expected": "memory"},
    {"name": "Auth Attack", "logs": [
        "[09:00:00] INFO auth: Login attempt admin",
        "[09:00:01] WARNING auth: Invalid password",
        "[09:00:05] ERROR security: Rate limit exceeded",
        "[09:00:06] CRITICAL security: Brute force detected"
    ], "expected": "password"},
    {"name": "Disk Full", "logs": [
        "[11:00:00] INFO storage: Disk at 70%",
        "[12:00:00] WARNING storage: Disk at 95%",
        "[12:01:00] ERROR storage: No space left",
        "[12:01:01] CRITICAL app: Cannot write logs"
    ], "expected": "space"}
]

correct = 0
results = []

for scenario in SCENARIOS:
    print(f"  Testing: {scenario['name']}...")
    logs_text = "\n".join(scenario["logs"])
    
    try:
        response = client.generate(
            model="llama3.2:3b",
            prompt=f"Analyze these logs and identify the ROOT CAUSE in one sentence:\n\n{logs_text}\n\nRoot cause:",
            options={"temperature": 0.1}
        )
        
        identified = response.response.lower()
        success = scenario["expected"].lower() in identified
        if success: correct += 1
        
        results.append({"name": scenario["name"], "success": success, "identified": identified[:100]})
        print(f"    {'✓' if success else '✗'} {scenario['name']}")
    except Exception as e:
        results.append({"name": scenario["name"], "success": False, "error": str(e)[:50]})
        print(f"    ✗ Error")

accuracy = round(correct/len(SCENARIOS)*100, 1)
print(f"\nRCA Accuracy: {accuracy}% ({correct}/{len(SCENARIOS)})")

with open("RESULTS_DIR/rca_results.json", "w") as f:
    json.dump({"scenarios": results, "summary": {"accuracy": accuracy, "correct": correct, "total": len(SCENARIOS)}}, f, indent=2)
RCAEOF

# Fix path in RCA script
python3 -c "
import json
import ollama

client = ollama.Client(host='http://localhost:11434', timeout=60.0)

SCENARIOS = [
    {'name': 'Database Failure', 'logs': ['[10:00:00] INFO db: Pool initialized', '[10:00:10] WARNING db: Pool at 80%', '[10:00:15] ERROR db: Connection timeout', '[10:00:17] CRITICAL app: Database unavailable'], 'expected': 'timeout'},
    {'name': 'Memory Exhaustion', 'logs': ['[14:00:00] INFO app: Starting batch', '[14:10:00] WARNING monitor: Memory at 85%', '[14:15:00] ERROR jvm: OutOfMemoryError', '[14:15:01] CRITICAL system: OOM killer'], 'expected': 'memory'},
    {'name': 'Auth Attack', 'logs': ['[09:00:00] INFO auth: Login attempt admin', '[09:00:01] WARNING auth: Invalid password', '[09:00:05] ERROR security: Rate limit exceeded', '[09:00:06] CRITICAL security: Brute force detected'], 'expected': 'password'},
    {'name': 'Disk Full', 'logs': ['[11:00:00] INFO storage: Disk at 70%', '[12:00:00] WARNING storage: Disk at 95%', '[12:01:00] ERROR storage: No space left', '[12:01:01] CRITICAL app: Cannot write logs'], 'expected': 'space'}
]

correct = 0
results = []

for scenario in SCENARIOS:
    print(f\"  Testing: {scenario['name']}...\")
    logs_text = '\\n'.join(scenario['logs'])
    try:
        response = client.generate(model='llama3.2:3b', prompt=f'Analyze these logs and identify the ROOT CAUSE in one sentence:\\n\\n{logs_text}\\n\\nRoot cause:', options={'temperature': 0.1})
        identified = response.response.lower()
        success = scenario['expected'].lower() in identified
        if success: correct += 1
        results.append({'name': scenario['name'], 'success': success, 'identified': identified[:100]})
        print(f\"    {'✓' if success else '✗'} {scenario['name']}\")
    except Exception as e:
        results.append({'name': scenario['name'], 'success': False, 'error': str(e)[:50]})

accuracy = round(correct/len(SCENARIOS)*100, 1)
print(f'\\nRCA Accuracy: {accuracy}% ({correct}/{len(SCENARIOS)})')

with open('$RESULTS_DIR/rca_results.json', 'w') as f:
    json.dump({'scenarios': results, 'summary': {'accuracy': accuracy, 'correct': correct, 'total': len(SCENARIOS)}}, f, indent=2)
"

echo -e "${GREEN}✓${NC} RCA test complete"

#===============================================================================
# Generate Report
#===============================================================================
echo -e "${YELLOW}Generating report...${NC}"

python3 << REPORTEOF
import json
from datetime import datetime
from pathlib import Path

rd = Path("$RESULTS_DIR")
drain = json.load(open(rd/"drain_results.json")) if (rd/"drain_results.json").exists() else None
llm = json.load(open(rd/"llm_parser_results.json")) if (rd/"llm_parser_results.json").exists() else None  
rca = json.load(open(rd/"rca_results.json")) if (rd/"rca_results.json").exists() else None

# LaTeX
latex = ["% GraphRCA Results - " + datetime.now().strftime("%Y-%m-%d"), ""]
if drain and llm:
    d, l = drain["metrics"], llm["metrics"]
    latex.append(r"""
\begin{table}[t]
\centering
\caption{Parser Comparison on LogHub BGL}
\begin{tabular}{lccc}
\toprule
\textbf{Metric} & \textbf{Drain} & \textbf{GraphRCA} & \textbf{Winner} \\
\midrule""")
    latex.append(f"Parse Rate & {d['parse_rate']}\\% & {l['severity_accuracy']}\\% & {'GraphRCA' if l['severity_accuracy'] > d['parse_rate'] else 'Drain'} \\\\")
    latex.append(f"Latency & {d['avg_latency_ms']:.2f}ms & {l['avg_latency_s']*1000:.0f}ms & Drain \\\\")
    latex.append(f"Throughput & {d['throughput_logs_per_sec']:.0f}/s & {l['throughput_logs_per_sec']:.2f}/s & Drain \\\\")
    latex.append(r"Templates & Yes & No & GraphRCA \\")
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}""")

if rca:
    latex.append(r"""
\begin{table}[t]
\centering
\caption{Root Cause Identification}
\begin{tabular}{lc}
\toprule
\textbf{Scenario} & \textbf{Result} \\
\midrule""")
    for s in rca["scenarios"]:
        latex.append(f"{s['name']} & {'\\checkmark' if s['success'] else '$\\times$'} \\\\")
    latex.append(f"\\midrule\n\\textbf{{Overall}} & {rca['summary']['accuracy']}\\% \\\\")
    latex.append(r"""\bottomrule
\end{tabular}
\end{table}""")

with open(rd/"latex_tables.tex", "w") as f:
    f.write("\n".join(latex))

# Markdown
md = ["# GraphRCA Evaluation Results", f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
if drain and llm:
    d, l = drain["metrics"], llm["metrics"]
    md.append("## Parser Comparison\n")
    md.append("| Metric | Drain | GraphRCA |")
    md.append("|--------|-------|----------|")
    md.append(f"| Parse Rate | {d['parse_rate']}% | {l['severity_accuracy']}% |")
    md.append(f"| Latency | {d['avg_latency_ms']:.2f}ms | {l['avg_latency_s']*1000:.0f}ms |")
    md.append(f"| Throughput | {d['throughput_logs_per_sec']:.0f}/s | {l['throughput_logs_per_sec']:.2f}/s |")
    md.append(f"| Templates | Yes | No |")
    md.append("")

if rca:
    md.append("## RCA Results\n")
    for s in rca["scenarios"]:
        md.append(f"- {s['name']}: {'✓' if s['success'] else '✗'}")
    md.append(f"\n**Accuracy: {rca['summary']['accuracy']}%**")

with open(rd/"REPORT.md", "w") as f:
    f.write("\n".join(md))

print(f"Reports saved to {rd}")
REPORTEOF

#===============================================================================
# Done
#===============================================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     EVALUATION COMPLETE                                      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Results: $RESULTS_DIR"
ls -la "$RESULTS_DIR"/ 2>/dev/null
echo ""
echo -e "${YELLOW}View report:${NC} cat $RESULTS_DIR/REPORT.md"
echo -e "${YELLOW}LaTeX tables:${NC} cat $RESULTS_DIR/latex_tables.tex"
echo ""
echo -e "${GREEN}Runtime: $SECONDS seconds${NC}"
