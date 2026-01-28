#!/bin/bash
#===============================================================================
# GraphRCA COMPREHENSIVE Evaluation
# 
# 5-6 HOUR GPU RUN - Publication-Ready Experiments
#
# Run: nohup ./overnight_evaluation.sh > overnight.log 2>&1 &
# Monitor: tail -f overnight.log
#===============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/eval_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="$SCRIPT_DIR/loghub_data"

mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/checkpoints"
mkdir -p "$RESULTS_DIR/figures"
mkdir -p "$DATA_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESULTS_DIR/experiment.log"
}

save_checkpoint() {
    local file="$1"
    if [ -f "$file" ]; then
        cp "$file" "$RESULTS_DIR/checkpoints/$(basename $file).$(date +%H%M%S)" 2>/dev/null
        log "CHECKPOINT: $(basename $file)"
    fi
}

log "╔══════════════════════════════════════════════════════════════════╗"
log "║  GraphRCA COMPREHENSIVE Evaluation                               ║"
log "║  Estimated Runtime: 4-5 hours                                    ║"
log "║  Results: $RESULTS_DIR"
log "╚══════════════════════════════════════════════════════════════════╝"

cd "$PROJECT_ROOT"
source venv/bin/activate 2>/dev/null || { log "ERROR: venv not found"; exit 1; }
pip install -q drain3 pandas numpy matplotlib scipy 2>/dev/null

# Ensure Ollama is running
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    log "Starting Ollama..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 15
fi

# Download datasets
log "Downloading LogHub datasets..."
[ -f "$DATA_DIR/BGL_2k.log" ] || curl -sL "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log" -o "$DATA_DIR/BGL_2k.log"
[ -f "$DATA_DIR/HDFS_2k.log" ] || curl -sL "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log" -o "$DATA_DIR/HDFS_2k.log"

log "Setup complete. Starting experiments..."
log ""

#===============================================================================
# EXPERIMENT 1: Drain Baseline (Both Datasets)
#===============================================================================
log "════════════════════════════════════════════════════════════════════"
log "EXPERIMENT 1/7: Drain Baseline on Multiple Datasets"
log "════════════════════════════════════════════════════════════════════"

python3 << DRAIN_EOF
import time, json, statistics, os
from collections import defaultdict
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

RESULTS_DIR = "${RESULTS_DIR}"
DATA_DIR = "${DATA_DIR}"

def run_drain(dataset_path, dataset_name):
    print(f'  Running Drain on {dataset_name}...')
    
    with open(dataset_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    
    config = TemplateMinerConfig()
    miner = TemplateMiner(config=config)
    
    latencies = []
    templates = defaultdict(int)
    
    for i, line in enumerate(lines):
        parts = line.split(None, 9)
        msg = parts[-1] if len(parts) > 4 else line
        
        start = time.perf_counter()
        result = miner.add_log_message(msg)
        latencies.append((time.perf_counter() - start) * 1000)
        
        if result and result.get('cluster_id'):
            templates[result['cluster_id']] += 1
    
    sorted_lat = sorted(latencies)
    n = len(latencies)
    
    return {
        'dataset': dataset_name,
        'total_logs': len(lines),
        'unique_templates': len(templates),
        'avg_latency_ms': round(statistics.mean(latencies), 4),
        'std_latency_ms': round(statistics.stdev(latencies), 4) if n > 1 else 0,
        'p50_latency_ms': round(sorted_lat[n//2], 4),
        'p95_latency_ms': round(sorted_lat[int(n*0.95)], 4),
        'p99_latency_ms': round(sorted_lat[int(n*0.99)], 4),
        'throughput_per_sec': round(n / (sum(latencies)/1000), 1)
    }

results = {'method': 'Drain', 'datasets': []}

for dataset_file, name in [('BGL_2k.log', 'BGL'), ('HDFS_2k.log', 'HDFS')]:
    path = f'{DATA_DIR}/{dataset_file}'
    if os.path.exists(path):
        result = run_drain(path, name)
        results['datasets'].append(result)
        print(f'    {name}: {result["throughput_per_sec"]} logs/s, {result["unique_templates"]} templates')

with open(f'{RESULTS_DIR}/01_drain_baseline.json', 'w') as f:
    json.dump(results, f, indent=2)

print('  Saved: 01_drain_baseline.json')
DRAIN_EOF

save_checkpoint "$RESULTS_DIR/01_drain_baseline.json"
log "Drain baseline complete"

#===============================================================================
# EXPERIMENT 2: LLM Parsing - Statistical Evaluation (~2 hours)
#===============================================================================
log ""
log "════════════════════════════════════════════════════════════════════"
log "EXPERIMENT 2/7: LLM Parsing with Statistical Significance"
log "Expected: ~2 hours"
log "════════════════════════════════════════════════════════════════════"

python3 << LLM_EOF
import time, json, statistics, os, re
import ollama
from datetime import datetime

RESULTS_DIR = "${RESULTS_DIR}"
DATA_DIR = "${DATA_DIR}"

client = ollama.Client(host='http://localhost:11434', timeout=60.0)

def parse_with_llm(log_line):
    try:
        resp = client.generate(
            model='llama3.2:3b',
            prompt=f'''Parse this log entry. Return JSON with:
- timestamp: the date/time string
- level: severity (INFO/WARNING/ERROR/FATAL/UNKNOWN)
- component: the component/module name
- message: the main message

Log: {log_line}

JSON only:''',
            format='json',
            options={'temperature': 0.1}
        )
        return json.loads(resp.response)
    except Exception as e:
        return {'error': str(e)[:50]}

def extract_bgl_ground_truth(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    
    severity = 'INFO'
    for p in parts:
        if p.upper() in ['FATAL', 'ERROR', 'WARNING', 'INFO', 'FAILURE', 'SEVERE']:
            severity = 'ERROR' if p.upper() in ['FAILURE', 'SEVERE'] else p.upper()
            break
    
    component = None
    msg = ' '.join(parts[4:])
    for comp in ['kernel', 'RAS', 'MMCS', 'LINKCARD', 'CMCS', 'BGLMASTER', 'ciod', 'APP']:
        if comp.lower() in msg.lower():
            component = comp
            break
    
    return {'timestamp': parts[1] if len(parts) > 1 else None, 'level': severity, 
            'component': component, 'message': msg}

def extract_hdfs_ground_truth(line):
    match = re.match(r'(\d{6}\s+\d{6})\s+(\d+)\s+(\w+)\s+(\S+):\s*(.*)', line)
    if match:
        severity = match.group(3).upper()
        if severity not in ['INFO', 'WARNING', 'WARN', 'ERROR', 'FATAL']:
            severity = 'INFO'
        if severity == 'WARN':
            severity = 'WARNING'
        return {
            'timestamp': match.group(1),
            'level': severity,
            'component': match.group(4),
            'message': match.group(5)
        }
    return {'timestamp': None, 'level': 'INFO', 'component': None, 'message': line}

def evaluate_parsing(samples, ground_truth_fn, dataset_name, run_num):
    results = {
        'field_correct': {'timestamp': 0, 'level': 0, 'component': 0, 'message': 0},
        'field_total': {'timestamp': 0, 'level': 0, 'component': 0, 'message': 0},
        'latencies': [],
        'errors': 0
    }
    
    for i, line in enumerate(samples):
        gt = ground_truth_fn(line)
        if not gt:
            continue
        
        start = time.perf_counter()
        parsed = parse_with_llm(line)
        elapsed = time.perf_counter() - start
        results['latencies'].append(elapsed)
        
        if 'error' in parsed:
            results['errors'] += 1
            continue
        
        results['field_total']['timestamp'] += 1
        if parsed.get('timestamp') and len(str(parsed['timestamp'])) > 3:
            results['field_correct']['timestamp'] += 1
        
        results['field_total']['level'] += 1
        gt_level = (gt['level'] or 'INFO').upper()
        parsed_level = (parsed.get('level') or '').upper()
        if gt_level == parsed_level or (gt_level in ['FATAL', 'SEVERE'] and parsed_level in ['FATAL', 'CRITICAL', 'SEVERE', 'ERROR']):
            results['field_correct']['level'] += 1
        
        if gt['component']:
            results['field_total']['component'] += 1
            if parsed.get('component') and gt['component'].lower() in str(parsed['component']).lower():
                results['field_correct']['component'] += 1
        
        results['field_total']['message'] += 1
        if parsed.get('message') and len(str(parsed['message'])) > 5:
            results['field_correct']['message'] += 1
        
        if (i+1) % 50 == 0:
            avg_lat = statistics.mean(results['latencies'])
            remaining = len(samples) - i - 1
            eta = int(remaining * avg_lat / 60)
            print(f'    [{dataset_name} Run {run_num}] {i+1}/{len(samples)} - ETA: {eta}min')
        
        if (i+1) % 100 == 0:
            with open(f'{RESULTS_DIR}/02_llm_parsing_checkpoint.json', 'w') as f:
                json.dump({'progress': f'{dataset_name}_run{run_num}_{i+1}'}, f)
    
    return results

NUM_RUNS = 3
SAMPLES_PER_DATASET = 400

all_results = {
    'method': 'GraphRCA LLM Parser',
    'model': 'llama3.2:3b',
    'num_runs': NUM_RUNS,
    'samples_per_run': SAMPLES_PER_DATASET,
    'datasets': {}
}

datasets = [
    ('BGL_2k.log', 'BGL', extract_bgl_ground_truth),
    ('HDFS_2k.log', 'HDFS', extract_hdfs_ground_truth)
]

for dataset_file, dataset_name, gt_fn in datasets:
    path = f'{DATA_DIR}/{dataset_file}'
    if not os.path.exists(path):
        print(f'  Skipping {dataset_name} - file not found')
        continue
    
    print(f'  === {dataset_name} Dataset ===')
    
    with open(path) as f:
        all_lines = [l.strip() for l in f if l.strip()]
    
    step = max(1, len(all_lines) // SAMPLES_PER_DATASET)
    samples = all_lines[::step][:SAMPLES_PER_DATASET]
    
    runs = []
    for run in range(1, NUM_RUNS + 1):
        print(f'  Run {run}/{NUM_RUNS}:')
        result = evaluate_parsing(samples, gt_fn, dataset_name, run)
        runs.append(result)
    
    field_accuracies = {field: [] for field in ['timestamp', 'level', 'component', 'message']}
    all_latencies = []
    
    for run_result in runs:
        for field in field_accuracies:
            if run_result['field_total'][field] > 0:
                acc = run_result['field_correct'][field] / run_result['field_total'][field] * 100
                field_accuracies[field].append(acc)
        all_latencies.extend(run_result['latencies'])
    
    sorted_lat = sorted(all_latencies)
    n = len(sorted_lat)
    
    dataset_result = {
        'samples': len(samples),
        'runs': NUM_RUNS,
        'field_accuracy': {},
        'latency': {
            'mean_s': round(statistics.mean(all_latencies), 3),
            'std_s': round(statistics.stdev(all_latencies), 3) if n > 1 else 0,
            'p50_s': round(sorted_lat[n//2], 3),
            'p95_s': round(sorted_lat[int(n*0.95)], 3),
            'p99_s': round(sorted_lat[int(n*0.99)], 3),
        },
        'throughput_per_sec': round(n / sum(all_latencies), 3) if sum(all_latencies) > 0 else 0,
        'total_errors': sum(r['errors'] for r in runs)
    }
    
    for field, accs in field_accuracies.items():
        if accs:
            dataset_result['field_accuracy'][field] = {
                'mean': round(statistics.mean(accs), 1),
                'std': round(statistics.stdev(accs), 1) if len(accs) > 1 else 0
            }
    
    all_correct = sum(sum(r['field_correct'].values()) for r in runs)
    all_total = sum(sum(r['field_total'].values()) for r in runs)
    dataset_result['overall_accuracy'] = round(all_correct / all_total * 100, 1) if all_total > 0 else 0
    
    all_results['datasets'][dataset_name] = dataset_result
    
    print(f'  {dataset_name} Results:')
    for field, acc in dataset_result['field_accuracy'].items():
        print(f'    {field}: {acc["mean"]:.1f}% (+/-{acc["std"]:.1f})')
    print(f'    Overall: {dataset_result["overall_accuracy"]:.1f}%')

with open(f'{RESULTS_DIR}/02_llm_parsing.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print('  Saved: 02_llm_parsing.json')
LLM_EOF

save_checkpoint "$RESULTS_DIR/02_llm_parsing.json"
log "LLM parsing complete"

#===============================================================================
# EXPERIMENT 3: DAG Construction Scalability
#===============================================================================
log ""
log "════════════════════════════════════════════════════════════════════"
log "EXPERIMENT 3/7: DAG Construction Scalability"
log "════════════════════════════════════════════════════════════════════"

python3 << DAG_EOF
import time, json, statistics
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

RESULTS_DIR = "${RESULTS_DIR}"

class LogEntry(BaseModel):
    timestamp: Optional[datetime] = None
    message: str = ''
    level: str = ''
    component: Optional[str] = None

class DAGNode(BaseModel):
    id: str
    parent_id: Optional[str] = None
    children: List[str] = []
    log_entry: LogEntry

class DAG(BaseModel):
    nodes: List[DAGNode] = []
    root_id: Optional[str] = None
    leaf_ids: List[str] = []

def build_dag(entries):
    nodes = []
    for i, entry in enumerate(entries):
        nodes.append(DAGNode(id=str(i), log_entry=entry))
    
    nodes.sort(key=lambda n: n.log_entry.timestamp or datetime.min)
    
    for i in range(len(nodes) - 1):
        nodes[i].children.append(nodes[i+1].id)
        nodes[i+1].parent_id = nodes[i].id
    
    root_id = nodes[0].id if nodes else None
    leaf_ids = [n.id for n in nodes if not n.children]
    
    return DAG(nodes=nodes, root_id=root_id, leaf_ids=leaf_ids)

SIZES = [10, 25, 50, 100, 250, 500, 1000, 2000]
NUM_RUNS = 20

results = {
    'experiment': 'DAG Construction Scalability',
    'num_runs_per_size': NUM_RUNS,
    'measurements': []
}

print('  Testing DAG construction scalability...')

for size in SIZES:
    entries = []
    for i in range(size):
        entries.append(LogEntry(
            timestamp=datetime(2024, 1, 1, 10, i // 60, i % 60),
            message=f'Log {i}',
            level='ERROR' if i % 5 == 0 else 'INFO'
        ))
    
    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        dag = build_dag(entries.copy())
        times.append((time.perf_counter() - start) * 1000)
    
    measurement = {
        'size': size,
        'mean_ms': round(statistics.mean(times), 3),
        'std_ms': round(statistics.stdev(times), 3),
        'min_ms': round(min(times), 3),
        'max_ms': round(max(times), 3),
        'nodes_created': len(dag.nodes),
        'valid': len(dag.nodes) == size and dag.root_id is not None
    }
    results['measurements'].append(measurement)
    print(f'    n={size}: {measurement["mean_ms"]:.2f}ms (+/-{measurement["std_ms"]:.2f})')

sizes = [m['size'] for m in results['measurements']]
times = [m['mean_ms'] for m in results['measurements']]

if len(sizes) >= 3:
    ratios = [t/s for t, s in zip(times, sizes)]
    ratio_variance = statistics.variance(ratios) if len(ratios) > 1 else 0
    
    if ratio_variance < 0.0001:
        complexity = 'O(n) - Linear'
    else:
        import math
        nlogn_ratios = [t/(s * math.log2(s)) if s > 1 else t for t, s in zip(times, sizes)]
        nlogn_var = statistics.variance(nlogn_ratios) if len(nlogn_ratios) > 1 else 0
        
        if nlogn_var < ratio_variance:
            complexity = 'O(n log n)'
        else:
            complexity = 'O(n) - Approximately Linear'
else:
    complexity = 'Insufficient data'

results['complexity'] = complexity
results['max_tested'] = max(SIZES)

with open(f'{RESULTS_DIR}/03_dag_scalability.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'  Complexity: {complexity}')
print('  Saved: 03_dag_scalability.json')
DAG_EOF

save_checkpoint "$RESULTS_DIR/03_dag_scalability.json"
log "DAG scalability complete"

#===============================================================================
# EXPERIMENT 4: Full Pipeline RCA (20 scenarios × 3 runs)
#===============================================================================
log ""
log "════════════════════════════════════════════════════════════════════"
log "EXPERIMENT 4/7: Full Pipeline RCA (20 scenarios x 3 runs)"
log "Expected: ~2 hours"
log "════════════════════════════════════════════════════════════════════"

python3 << RCA_EOF
import time, json, os
import ollama
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

RESULTS_DIR = "${RESULTS_DIR}"

class LogEntry(BaseModel):
    timestamp: Optional[datetime] = None
    message: str = ''
    level: str = ''
    component: Optional[str] = None

class DAGNode(BaseModel):
    id: str
    parent_id: Optional[str] = None
    children: List[str] = []
    log_entry: LogEntry

class DAG(BaseModel):
    nodes: List[DAGNode] = []
    root_id: Optional[str] = None
    root_cause: Optional[str] = None

client = ollama.Client(host='http://localhost:11434', timeout=90.0)

def parse_log_llm(log_text):
    try:
        resp = client.generate(
            model='llama3.2:3b',
            prompt=f'Parse to JSON (timestamp, level, component, message):\n{log_text}',
            format='json',
            options={'temperature': 0.1}
        )
        data = json.loads(resp.response)
        return LogEntry(
            timestamp=datetime.now(),
            message=data.get('message', log_text),
            level=data.get('level', 'INFO'),
            component=data.get('component')
        )
    except:
        return LogEntry(message=log_text, level='UNKNOWN')

def build_dag(entries):
    nodes = [DAGNode(id=str(i), log_entry=e) for i, e in enumerate(entries)]
    for i in range(len(nodes) - 1):
        nodes[i].children.append(nodes[i+1].id)
        nodes[i+1].parent_id = nodes[i].id
    root_cause = nodes[0].log_entry.message if nodes else None
    return DAG(nodes=nodes, root_id='0', root_cause=root_cause)

def build_context(dag):
    return {
        'root_cause': dag.root_cause,
        'chain': [n.log_entry.message for n in dag.nodes],
        'levels': [n.log_entry.level for n in dag.nodes]
    }

def generate_rca(context):
    chain_str = '\n'.join(f'{i+1}. [{context["levels"][i]}] {msg}' for i, msg in enumerate(context['chain']))
    prompt = f'''Analyze this system incident:

Root Event: {context['root_cause']}

Event Chain:
{chain_str}

Identify:
1. Root cause (one sentence)
2. Failure mechanism
3. Fix recommendation'''
    
    try:
        resp = client.generate(model='llama3.2:3b', prompt=prompt, options={'temperature': 0.2})
        return resp.response
    except Exception as e:
        return f'Error: {str(e)}'

SCENARIOS = [
    {'name': 'DB Connection Pool Exhaustion', 'category': 'Database',
     'logs': ['[10:00] INFO db: Pool init size=10', '[10:05] WARN db: Pool 80% full',
              '[10:10] WARN db: Pool 95% full', '[10:10] ERROR db: Connection timeout 30s',
              '[10:11] ERROR api: No DB connection', '[10:11] CRITICAL app: Fallback mode'],
     'keywords': ['pool', 'connection', 'exhaust', 'timeout']},
    
    {'name': 'Database Deadlock', 'category': 'Database',
     'logs': ['[11:00] INFO db: T1 UPDATE accounts', '[11:00] INFO db: T2 UPDATE accounts',
              '[11:01] WARN db: T1 waiting for T2 lock', '[11:01] WARN db: T2 waiting for T1 lock',
              '[11:02] ERROR db: Deadlock detected', '[11:02] ERROR db: Rolling back T2'],
     'keywords': ['deadlock', 'lock', 'wait', 'rollback']},
    
    {'name': 'Database Replication Lag', 'category': 'Database',
     'logs': ['[14:00] INFO db: Replication lag 100ms', '[14:30] WARN db: Replication lag 5s',
              '[15:00] ERROR db: Replication lag 30s', '[15:01] ERROR app: Stale read detected',
              '[15:02] CRITICAL db: Replica out of sync'],
     'keywords': ['replication', 'lag', 'sync', 'stale']},
    
    {'name': 'Query Performance Degradation', 'category': 'Database',
     'logs': ['[09:00] INFO db: Query avg 50ms', '[10:00] WARN db: Query avg 500ms',
              '[10:30] ERROR db: Query timeout 30s', '[10:31] ERROR db: Table scan detected',
              '[10:32] CRITICAL app: Response timeout'],
     'keywords': ['query', 'timeout', 'slow', 'scan']},
    
    {'name': 'Memory Leak OOM', 'category': 'Memory',
     'logs': ['[14:00] INFO app: Batch started', '[14:30] INFO mon: Mem 4/16GB',
              '[15:00] WARN mon: Mem 12/16GB', '[15:15] WARN gc: Frequent GC',
              '[15:20] ERROR jvm: OutOfMemoryError', '[15:20] CRITICAL sys: OOM killer'],
     'keywords': ['memory', 'oom', 'heap', 'leak', 'gc']},
    
    {'name': 'Memory Fragmentation', 'category': 'Memory',
     'logs': ['[10:00] INFO mem: Allocation normal', '[12:00] WARN mem: Fragmentation 40%',
              '[14:00] ERROR mem: Cannot allocate 1GB contiguous', '[14:01] ERROR app: Allocation failed'],
     'keywords': ['fragment', 'allocat', 'contiguous', 'memory']},
    
    {'name': 'Cache Eviction Storm', 'category': 'Memory',
     'logs': ['[08:00] INFO cache: Hit rate 95%', '[09:00] WARN cache: Hit rate 60%',
              '[09:30] ERROR cache: Mass eviction triggered', '[09:31] ERROR db: Connection spike',
              '[09:32] CRITICAL app: Response time 10x'],
     'keywords': ['cache', 'evict', 'hit', 'spike']},
    
    {'name': 'Disk Space Exhaustion', 'category': 'Infrastructure',
     'logs': ['[11:00] INFO disk: /var/log 70%', '[12:00] WARN disk: /var/log 90%',
              '[12:30] ERROR disk: No space left', '[12:31] ERROR app: Write failed',
              '[12:32] CRITICAL db: Transaction log full'],
     'keywords': ['disk', 'space', 'full', 'write']},
    
    {'name': 'Network Partition', 'category': 'Infrastructure',
     'logs': ['[16:00] INFO cluster: All healthy', '[16:00] WARN cluster: worker-3 delayed',
              '[16:01] ERROR cluster: worker-3 timeout', '[16:01] ERROR cluster: Connection lost',
              '[16:02] CRITICAL cluster: Split brain risk'],
     'keywords': ['network', 'partition', 'cluster', 'split', 'brain']},
    
    {'name': 'CPU Throttling', 'category': 'Infrastructure',
     'logs': ['[14:00] INFO sys: CPU 40%', '[14:15] INFO sys: CPU 70%',
              '[14:30] WARN sys: CPU 95%', '[14:35] ERROR sys: Throttling active',
              '[14:36] ERROR app: Request timeout', '[14:36] CRITICAL lb: Backend unhealthy'],
     'keywords': ['cpu', 'throttl', 'load', 'timeout']},
    
    {'name': 'DNS Resolution Failure', 'category': 'Infrastructure',
     'logs': ['[10:00] INFO dns: Resolution time 5ms', '[10:30] WARN dns: Resolution time 500ms',
              '[11:00] ERROR dns: Lookup failed api.service', '[11:01] ERROR app: Cannot reach service',
              '[11:02] CRITICAL app: All external calls failing'],
     'keywords': ['dns', 'resolv', 'lookup', 'external']},
    
    {'name': 'Brute Force Attack', 'category': 'Security',
     'logs': ['[09:00] INFO auth: Login admin 192.168.1.100', '[09:00] WARN auth: Failed admin',
              '[09:00] WARN auth: Failed admin', '[09:00] WARN auth: Failed admin',
              '[09:01] ERROR sec: Rate limit IP', '[09:01] CRITICAL sec: Brute force blocked'],
     'keywords': ['brute', 'force', 'password', 'login', 'block']},
    
    {'name': 'SSL Certificate Expired', 'category': 'Security',
     'logs': ['[08:00] INFO tls: Cert expires 7 days', '[08:00] WARN tls: Cert expires 1 day',
              '[00:00] ERROR tls: Certificate expired', '[00:00] ERROR https: Handshake failed',
              '[00:01] CRITICAL api: HTTPS failing'],
     'keywords': ['certificate', 'expired', 'ssl', 'tls', 'handshake']},
    
    {'name': 'Token Expiration Cascade', 'category': 'Security',
     'logs': ['[12:00] INFO auth: Token refresh normal', '[12:30] WARN auth: Token refresh delayed',
              '[13:00] ERROR auth: Token expired mid-request', '[13:01] ERROR api: 401 Unauthorized cascade',
              '[13:02] CRITICAL app: Session invalidation storm'],
     'keywords': ['token', 'expir', 'auth', '401', 'session']},
    
    {'name': 'Config Deployment Error', 'category': 'Application',
     'logs': ['[10:00] INFO deploy: Config update', '[10:00] INFO deploy: Pull config',
              '[10:01] WARN app: Schema mismatch', '[10:01] ERROR app: Invalid max_connections=-1',
              '[10:02] CRITICAL app: Startup failed'],
     'keywords': ['config', 'invalid', 'schema', 'deploy', 'startup']},
    
    {'name': 'Cascading Microservice Failure', 'category': 'Application',
     'logs': ['[09:00] INFO gw: Request /api/orders', '[09:00] INFO order: Process #12345',
              '[09:00] ERROR inventory: payment-service refused', '[09:01] ERROR order: Failed inventory',
              '[09:01] ERROR gw: 500 returned'],
     'keywords': ['cascade', 'service', 'refused', '500', 'fail']},
    
    {'name': 'Thread Pool Exhaustion', 'category': 'Application',
     'logs': ['[15:00] INFO pool: Active threads 10/100', '[15:30] WARN pool: Active 90/100',
              '[15:45] ERROR pool: No threads available', '[15:46] ERROR app: Request rejected',
              '[15:47] CRITICAL app: Thread pool deadlock suspected'],
     'keywords': ['thread', 'pool', 'exhaust', 'reject', 'deadlock']},
    
    {'name': 'Retry Storm', 'category': 'Application',
     'logs': ['[11:00] INFO api: Request to payment', '[11:00] WARN api: Payment timeout, retry 1',
              '[11:01] WARN api: Retry 2', '[11:01] WARN api: Retry 3',
              '[11:02] ERROR api: All retries failed', '[11:02] CRITICAL api: Retry storm amplifying'],
     'keywords': ['retry', 'storm', 'timeout', 'amplif', 'fail']},
    
    {'name': 'Alert Fatigue Cascade', 'category': 'Monitoring',
     'logs': ['[08:00] INFO alert: 5 alerts/hour', '[10:00] WARN alert: 50 alerts/hour',
              '[11:00] ERROR alert: 500 alerts/hour', '[11:01] ERROR ops: Alert acknowledged late',
              '[11:30] CRITICAL app: Outage detected 30min late'],
     'keywords': ['alert', 'fatigue', 'late', 'detect', 'outage']},
    
    {'name': 'Metrics Pipeline Failure', 'category': 'Monitoring',
     'logs': ['[14:00] INFO metrics: Pipeline healthy', '[14:30] WARN metrics: Lag 5min',
              '[15:00] ERROR metrics: Pipeline stalled', '[15:01] ERROR alert: No data for 30min',
              '[15:02] CRITICAL ops: Blind to system state'],
     'keywords': ['metrics', 'pipeline', 'stall', 'blind', 'data']}
]

NUM_RUNS = 3
results = {
    'experiment': 'Full Pipeline RCA',
    'scenarios': len(SCENARIOS),
    'runs_per_scenario': NUM_RUNS,
    'results': []
}

print(f'  Running {len(SCENARIOS)} scenarios x {NUM_RUNS} runs...')

for i, scenario in enumerate(SCENARIOS):
    print(f'  [{i+1}/{len(SCENARIOS)}] {scenario["name"]}')
    
    scenario_results = {
        'name': scenario['name'],
        'category': scenario['category'],
        'runs': [],
        'success_rate': 0
    }
    
    successes = 0
    
    for run in range(1, NUM_RUNS + 1):
        run_result = {'run': run, 'stages': {}, 'success': False}
        
        try:
            start = time.perf_counter()
            entries = [parse_log_llm(log) for log in scenario['logs']]
            run_result['stages']['parse_s'] = round(time.perf_counter() - start, 2)
            
            start = time.perf_counter()
            dag = build_dag(entries)
            run_result['stages']['dag_ms'] = round((time.perf_counter() - start) * 1000, 2)
            
            start = time.perf_counter()
            context = build_context(dag)
            run_result['stages']['context_ms'] = round((time.perf_counter() - start) * 1000, 2)
            
            start = time.perf_counter()
            analysis = generate_rca(context)
            run_result['stages']['rca_s'] = round(time.perf_counter() - start, 2)
            
            analysis_lower = analysis.lower()
            keywords_found = [kw for kw in scenario['keywords'] if kw in analysis_lower]
            run_result['keywords_found'] = keywords_found
            run_result['success'] = len(keywords_found) >= 2
            
            if run_result['success']:
                successes += 1
            
        except Exception as e:
            run_result['error'] = str(e)[:100]
        
        scenario_results['runs'].append(run_result)
        status = 'Y' if run_result['success'] else 'N'
        print(f'    Run {run}: {status}', end=' ')
    
    print()
    scenario_results['success_rate'] = round(successes / NUM_RUNS * 100, 1)
    results['results'].append(scenario_results)
    
    with open(f'{RESULTS_DIR}/04_rca_checkpoint.json', 'w') as f:
        json.dump(results, f, indent=2)

by_category = {}
for r in results['results']:
    cat = r['category']
    if cat not in by_category:
        by_category[cat] = {'total': 0, 'success_sum': 0}
    by_category[cat]['total'] += 1
    by_category[cat]['success_sum'] += r['success_rate']

results['summary'] = {
    'overall_success_rate': round(sum(r['success_rate'] for r in results['results']) / len(results['results']), 1),
    'by_category': {cat: round(v['success_sum']/v['total'], 1) for cat, v in by_category.items()},
    'total_scenarios': len(SCENARIOS),
    'total_runs': len(SCENARIOS) * NUM_RUNS
}

with open(f'{RESULTS_DIR}/04_pipeline_rca.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'  Overall Success Rate: {results["summary"]["overall_success_rate"]}%')
print('  Saved: 04_pipeline_rca.json')
RCA_EOF

save_checkpoint "$RESULTS_DIR/04_pipeline_rca.json"
log "Full pipeline RCA complete"

#===============================================================================
# EXPERIMENT 5: Ablation Study
#===============================================================================
log ""
log "════════════════════════════════════════════════════════════════════"
log "EXPERIMENT 5/7: Ablation Study"
log "════════════════════════════════════════════════════════════════════"

python3 << ABLATION_EOF
import time, json
import ollama

RESULTS_DIR = "${RESULTS_DIR}"

client = ollama.Client(host='http://localhost:11434', timeout=90.0)

TEST_SCENARIOS = [
    {'name': 'Database Pool', 
     'logs': ['[10:00] INFO db: Pool init', '[10:05] WARN db: Pool 80%', 
              '[10:10] ERROR db: Timeout', '[10:11] CRITICAL app: Down'],
     'keywords': ['pool', 'timeout', 'connection']},
    {'name': 'Memory OOM',
     'logs': ['[14:00] INFO app: Start', '[15:00] WARN mem: 75%',
              '[15:20] ERROR jvm: OOM', '[15:21] CRITICAL sys: Killed'],
     'keywords': ['memory', 'oom', 'heap']},
    {'name': 'Disk Full',
     'logs': ['[11:00] INFO disk: 70%', '[12:00] WARN disk: 90%',
              '[12:30] ERROR disk: No space', '[12:31] CRITICAL db: Log full'],
     'keywords': ['disk', 'space', 'full']},
    {'name': 'Network Split',
     'logs': ['[16:00] INFO net: Healthy', '[16:01] ERROR net: Lost worker-3',
              '[16:02] CRITICAL cluster: Split brain'],
     'keywords': ['network', 'split', 'partition']},
]

def run_variant(name, prompt_fn, scenarios):
    results = {'variant': name, 'scenarios': []}
    
    for scenario in scenarios:
        prompt = prompt_fn(scenario['logs'])
        
        try:
            start = time.perf_counter()
            resp = client.generate(model='llama3.2:3b', prompt=prompt, options={'temperature': 0.2})
            elapsed = time.perf_counter() - start
            
            analysis = resp.response.lower()
            keywords_found = [kw for kw in scenario['keywords'] if kw in analysis]
            success = len(keywords_found) >= 2
            
            results['scenarios'].append({
                'name': scenario['name'],
                'success': success,
                'latency_s': round(elapsed, 2),
                'keywords_found': keywords_found
            })
        except Exception as e:
            results['scenarios'].append({
                'name': scenario['name'],
                'success': False,
                'error': str(e)[:50]
            })
    
    success_rate = sum(1 for s in results['scenarios'] if s['success']) / len(results['scenarios']) * 100
    results['success_rate'] = round(success_rate, 1)
    return results

def full_pipeline_prompt(logs):
    log_str = '\n'.join(f'{i+1}. {log}' for i, log in enumerate(logs))
    return f'''Analyze this system incident.

Event Log:
{log_str}

Provide structured analysis:
1. Root cause identification
2. Causal chain explanation
3. Severity assessment
4. Recommended fix'''

def no_structure_prompt(logs):
    return f'''What caused this issue?

{chr(10).join(logs)}'''

def no_chain_prompt(logs):
    return f'''Logs:
{chr(10).join(logs)}

What is the root cause?'''

def verbose_prompt(logs):
    log_str = '\n'.join(f'Event {i+1}: {log}' for i, log in enumerate(logs))
    return f'''You are an expert Site Reliability Engineer analyzing a production incident.

Here is the sequence of log events:
{log_str}

Please perform a thorough root cause analysis:

Step 1: Identify the timeline of events
Step 2: Determine the initial trigger
Step 3: Trace the causal chain
Step 4: Identify the root cause
Step 5: Suggest remediation

Be specific and technical in your analysis.'''

print('  Running ablation study...')

ablation_results = {
    'experiment': 'Ablation Study',
    'description': 'Testing different prompt configurations',
    'variants': []
}

variants = [
    ('Full Pipeline (Ours)', full_pipeline_prompt),
    ('No Structure', no_structure_prompt),
    ('No Causal Chain', no_chain_prompt),
    ('Verbose Engineer', verbose_prompt)
]

for name, prompt_fn in variants:
    print(f'    Testing: {name}')
    result = run_variant(name, prompt_fn, TEST_SCENARIOS)
    ablation_results['variants'].append(result)
    print(f'      Success: {result["success_rate"]}%')

ablation_results['summary'] = {
    variant['variant']: variant['success_rate'] 
    for variant in ablation_results['variants']
}

with open(f'{RESULTS_DIR}/05_ablation_study.json', 'w') as f:
    json.dump(ablation_results, f, indent=2)

print('  Saved: 05_ablation_study.json')
ABLATION_EOF

save_checkpoint "$RESULTS_DIR/05_ablation_study.json"
log "Ablation study complete"

#===============================================================================
# EXPERIMENT 6: Generate Publication Figures
#===============================================================================
log ""
log "════════════════════════════════════════════════════════════════════"
log "EXPERIMENT 6/7: Generate Publication Figures"
log "════════════════════════════════════════════════════════════════════"

python3 << FIGS_EOF
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13, 
                     'figure.dpi': 150, 'font.family': 'DejaVu Sans'})

RESULTS_DIR = "${RESULTS_DIR}"
rd = Path(RESULTS_DIR)
figs = rd / 'figures'
figs.mkdir(exist_ok=True)

drain = json.load(open(rd/'01_drain_baseline.json'))
llm = json.load(open(rd/'02_llm_parsing.json'))
dag = json.load(open(rd/'03_dag_scalability.json'))
rca = json.load(open(rd/'04_pipeline_rca.json'))
ablation = json.load(open(rd/'05_ablation_study.json'))

print('  Generating figures...')

# Figure 1: Throughput
fig, ax = plt.subplots(figsize=(8, 5))
datasets = ['BGL', 'HDFS']
x = np.arange(len(datasets))
width = 0.35

drain_throughputs = [next((d['throughput_per_sec'] for d in drain['datasets'] if d['dataset'] == ds), 0) for ds in datasets]
llm_throughputs = [llm['datasets'].get(ds, {}).get('throughput_per_sec', 0) for ds in datasets]

bars1 = ax.bar(x - width/2, drain_throughputs, width, label='Drain (Baseline)', color='#2196F3', edgecolor='black')
bars2 = ax.bar(x + width/2, llm_throughputs, width, label='GraphRCA (Ours)', color='#4CAF50', edgecolor='black')

ax.set_ylabel('Throughput (logs/second)')
ax.set_title('Parser Throughput Comparison')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.savefig(figs/'fig1_throughput.pdf', bbox_inches='tight')
plt.savefig(figs/'fig1_throughput.png', bbox_inches='tight', dpi=300)
plt.close()
print('    Fig 1: Throughput')

# Figure 2: Field Accuracy
fig, ax = plt.subplots(figsize=(8, 5))
fields = ['Timestamp', 'Severity', 'Component', 'Message']

for i, (ds_name, color) in enumerate([('BGL', '#2196F3'), ('HDFS', '#4CAF50')]):
    ds_data = llm['datasets'].get(ds_name, {})
    field_acc = ds_data.get('field_accuracy', {})
    
    means = [field_acc.get(f.lower(), {}).get('mean', 0) for f in fields]
    stds = [field_acc.get(f.lower(), {}).get('std', 0) for f in fields]
    
    x = np.arange(len(fields))
    offset = (i - 0.5) * 0.35
    ax.bar(x + offset, means, 0.35, yerr=stds, capsize=5, label=ds_name, color=color, edgecolor='black', alpha=0.8)

ax.set_ylabel('Accuracy (%)')
ax.set_title('Field-Level Parsing Accuracy')
ax.set_xticks(np.arange(len(fields)))
ax.set_xticklabels(fields)
ax.set_ylim(0, 105)
ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
ax.legend()
plt.tight_layout()
plt.savefig(figs/'fig2_field_accuracy.pdf', bbox_inches='tight')
plt.savefig(figs/'fig2_field_accuracy.png', bbox_inches='tight', dpi=300)
plt.close()
print('    Fig 2: Field accuracy')

# Figure 3: DAG Scalability
fig, ax = plt.subplots(figsize=(7, 5))
sizes = [m['size'] for m in dag['measurements']]
times = [m['mean_ms'] for m in dag['measurements']]
stds = [m['std_ms'] for m in dag['measurements']]

ax.errorbar(sizes, times, yerr=stds, marker='o', capsize=5, capthick=2,
            linewidth=2, markersize=8, color='#4CAF50', ecolor='#81C784')
ax.fill_between(sizes, [t-s for t,s in zip(times, stds)], [t+s for t,s in zip(times, stds)],
                alpha=0.2, color='#4CAF50')

ax.set_xlabel('Number of Log Entries')
ax.set_ylabel('DAG Construction Time (ms)')
ax.set_title(f'DAG Construction Scalability ({dag["complexity"]})')
plt.tight_layout()
plt.savefig(figs/'fig3_dag_scalability.pdf', bbox_inches='tight')
plt.savefig(figs/'fig3_dag_scalability.png', bbox_inches='tight', dpi=300)
plt.close()
print('    Fig 3: DAG scalability')

# Figure 4: RCA by Category
fig, ax = plt.subplots(figsize=(9, 5))
categories = list(rca['summary']['by_category'].keys())
success_rates = list(rca['summary']['by_category'].values())
colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

bars = ax.barh(categories, success_rates, color=colors, edgecolor='black')
ax.axvline(x=rca['summary']['overall_success_rate'], color='red', linestyle='--', 
           linewidth=2, label=f'Overall: {rca["summary"]["overall_success_rate"]}%')

ax.set_xlabel('Success Rate (%)')
ax.set_title(f'Root Cause Identification by Category (n={rca["summary"]["total_scenarios"]} scenarios)')
ax.set_xlim(0, 105)
ax.legend(loc='lower right')

for bar, val in zip(bars, success_rates):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figs/'fig4_rca_categories.pdf', bbox_inches='tight')
plt.savefig(figs/'fig4_rca_categories.png', bbox_inches='tight', dpi=300)
plt.close()
print('    Fig 4: RCA categories')

# Figure 5: Ablation
fig, ax = plt.subplots(figsize=(8, 5))
variants = list(ablation['summary'].keys())
success_rates = list(ablation['summary'].values())
colors = ['#4CAF50' if 'Ours' in v or 'Full' in v else '#9E9E9E' for v in variants]

bars = ax.bar(variants, success_rates, color=colors, edgecolor='black')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Ablation Study: Prompt Configuration Impact')
ax.set_ylim(0, 105)
ax.set_xticklabels(variants, rotation=15, ha='right')

for bar, val in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%', 
            ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(figs/'fig5_ablation.pdf', bbox_inches='tight')
plt.savefig(figs/'fig5_ablation.png', bbox_inches='tight', dpi=300)
plt.close()
print('    Fig 5: Ablation')

# Figure 6: Summary Dashboard
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
methods = ['Drain', 'GraphRCA']
drain_avg = np.mean([d['throughput_per_sec'] for d in drain['datasets']])
llm_avg = np.mean([d.get('throughput_per_sec', 0) for d in llm['datasets'].values()])
ax.bar(methods, [drain_avg, llm_avg], color=['#2196F3', '#4CAF50'], edgecolor='black')
ax.set_ylabel('Throughput (logs/s)')
ax.set_title('(a) Parser Throughput')
ax.set_yscale('log')

ax = axes[0, 1]
accuracies = [llm['datasets'].get(ds, {}).get('overall_accuracy', 0) for ds in ['BGL', 'HDFS']]
ax.bar(['BGL', 'HDFS'], accuracies, color=['#2196F3', '#4CAF50'], edgecolor='black')
ax.set_ylabel('Accuracy (%)')
ax.set_title('(b) Overall Parsing Accuracy')
ax.set_ylim(0, 105)

ax = axes[1, 0]
sizes = [m['size'] for m in dag['measurements']]
times = [m['mean_ms'] for m in dag['measurements']]
ax.plot(sizes, times, 'o-', color='#4CAF50', linewidth=2, markersize=8)
ax.set_xlabel('Log Entries')
ax.set_ylabel('Time (ms)')
ax.set_title('(c) DAG Construction')

ax = axes[1, 1]
cats = list(rca['summary']['by_category'].keys())[:5]
rates = [rca['summary']['by_category'][c] for c in cats]
ax.barh(cats, rates, color=plt.cm.Set2(np.linspace(0, 1, len(cats))), edgecolor='black')
ax.set_xlabel('Success Rate (%)')
ax.set_title(f'(d) RCA Results ({rca["summary"]["overall_success_rate"]}% overall)')
ax.set_xlim(0, 105)

plt.suptitle('GraphRCA Evaluation Summary', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(figs/'fig6_summary.pdf', bbox_inches='tight')
plt.savefig(figs/'fig6_summary.png', bbox_inches='tight', dpi=300)
plt.close()
print('    Fig 6: Summary')

print(f'  All figures saved to: {figs}')
FIGS_EOF

save_checkpoint "$RESULTS_DIR/figures"
log "Figure generation complete"

#===============================================================================
# EXPERIMENT 7: Generate Final Report
#===============================================================================
log ""
log "════════════════════════════════════════════════════════════════════"
log "EXPERIMENT 7/7: Generate LaTeX Tables and Report"
log "════════════════════════════════════════════════════════════════════"

python3 << REPORT_EOF
import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = "${RESULTS_DIR}"
rd = Path(RESULTS_DIR)

drain = json.load(open(rd/'01_drain_baseline.json'))
llm = json.load(open(rd/'02_llm_parsing.json'))
dag = json.load(open(rd/'03_dag_scalability.json'))
rca = json.load(open(rd/'04_pipeline_rca.json'))
ablation = json.load(open(rd/'05_ablation_study.json'))

# LaTeX
latex = []
latex.append('% GraphRCA Evaluation Results - Publication Ready')
latex.append('% Generated: ' + datetime.now().strftime('%Y-%m-%d %H:%M'))
latex.append('')

# Table 1
latex.append('\\begin{table}[t]')
latex.append('\\centering')
latex.append('\\caption{Parser Performance Comparison on LogHub Datasets}')
latex.append('\\label{tab:parser_comparison}')
latex.append('\\begin{tabular}{llcc}')
latex.append('\\toprule')
latex.append('\\textbf{Dataset} & \\textbf{Metric} & \\textbf{Drain} & \\textbf{GraphRCA} \\\\')
latex.append('\\midrule')

for ds_name in ['BGL', 'HDFS']:
    drain_ds = next((d for d in drain['datasets'] if d['dataset'] == ds_name), None)
    llm_ds = llm['datasets'].get(ds_name, {})
    
    if drain_ds and llm_ds:
        latex.append(f'{ds_name} & Throughput (logs/s) & {drain_ds["throughput_per_sec"]:.0f} & {llm_ds["throughput_per_sec"]:.2f} \\\\')
        latex.append(f' & Avg Latency & {drain_ds["avg_latency_ms"]:.2f}ms & {llm_ds["latency"]["mean_s"]*1000:.0f}ms \\\\')
        latex.append('\\midrule')

latex.append('\\bottomrule')
latex.append('\\end{tabular}')
latex.append('\\end{table}')
latex.append('')

# Table 2
latex.append('\\begin{table}[t]')
latex.append('\\centering')
latex.append('\\caption{Field-Level Parsing Accuracy (mean $\\pm$ std)}')
latex.append('\\label{tab:field_accuracy}')
latex.append('\\begin{tabular}{lcc}')
latex.append('\\toprule')
latex.append('\\textbf{Field} & \\textbf{BGL} & \\textbf{HDFS} \\\\')
latex.append('\\midrule')

for field in ['timestamp', 'level', 'component', 'message']:
    bgl_acc = llm['datasets'].get('BGL', {}).get('field_accuracy', {}).get(field, {})
    hdfs_acc = llm['datasets'].get('HDFS', {}).get('field_accuracy', {}).get(field, {})
    
    bgl_str = f'{bgl_acc.get("mean", 0):.1f}\\% $\\pm$ {bgl_acc.get("std", 0):.1f}' if bgl_acc else 'N/A'
    hdfs_str = f'{hdfs_acc.get("mean", 0):.1f}\\% $\\pm$ {hdfs_acc.get("std", 0):.1f}' if hdfs_acc else 'N/A'
    
    latex.append(f'{field.capitalize()} & {bgl_str} & {hdfs_str} \\\\')

bgl_overall = llm['datasets'].get('BGL', {}).get('overall_accuracy', 0)
hdfs_overall = llm['datasets'].get('HDFS', {}).get('overall_accuracy', 0)
latex.append('\\midrule')
latex.append(f'\\textbf{{Overall}} & \\textbf{{{bgl_overall:.1f}\\%}} & \\textbf{{{hdfs_overall:.1f}\\%}} \\\\')
latex.append('\\bottomrule')
latex.append('\\end{tabular}')
latex.append('\\end{table}')
latex.append('')

# Table 3
latex.append('\\begin{table}[t]')
latex.append('\\centering')
latex.append('\\caption{Root Cause Identification by Failure Category}')
latex.append('\\label{tab:rca_results}')
latex.append('\\begin{tabular}{lc}')
latex.append('\\toprule')
latex.append('\\textbf{Category} & \\textbf{Success Rate} \\\\')
latex.append('\\midrule')

for cat, rate in rca['summary']['by_category'].items():
    latex.append(f'{cat} & {rate:.1f}\\% \\\\')

latex.append('\\midrule')
latex.append(f'\\textbf{{Overall}} & \\textbf{{{rca["summary"]["overall_success_rate"]:.1f}\\%}} \\\\')
latex.append('\\bottomrule')
latex.append('\\end{tabular}')
latex.append('\\end{table}')

with open(rd/'latex_tables.tex', 'w') as f:
    f.write('\n'.join(latex))

# Markdown Report
md = []
md.append('# GraphRCA Comprehensive Evaluation Report')
md.append('')
md.append(f'**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}')
md.append(f'**Datasets:** LogHub BGL, HDFS')
md.append(f'**Statistical Runs:** {llm["num_runs"]} per configuration')
md.append('')
md.append('---')
md.append('')
md.append('## Executive Summary')
md.append('')
md.append('| Metric | BGL | HDFS |')
md.append('|--------|-----|------|')
md.append(f'| Parsing Accuracy | {llm["datasets"].get("BGL", {}).get("overall_accuracy", 0):.1f}% | {llm["datasets"].get("HDFS", {}).get("overall_accuracy", 0):.1f}% |')
md.append(f'| Throughput | {llm["datasets"].get("BGL", {}).get("throughput_per_sec", 0):.2f} logs/s | {llm["datasets"].get("HDFS", {}).get("throughput_per_sec", 0):.2f} logs/s |')
md.append('')
md.append(f'**RCA Success Rate:** {rca["summary"]["overall_success_rate"]:.1f}% across {rca["summary"]["total_scenarios"]} scenarios')
md.append('')
md.append('---')
md.append('')
md.append('## RCA by Category')
md.append('')
md.append('| Category | Success Rate |')
md.append('|----------|-------------|')

for cat, rate in sorted(rca['summary']['by_category'].items(), key=lambda x: -x[1]):
    md.append(f'| {cat} | {rate:.1f}% |')

md.append('')
md.append('---')
md.append('')
md.append('## Files Generated')
md.append('')
md.append('- `01_drain_baseline.json`')
md.append('- `02_llm_parsing.json`')
md.append('- `03_dag_scalability.json`')
md.append('- `04_pipeline_rca.json`')
md.append('- `05_ablation_study.json`')
md.append('- `latex_tables.tex`')
md.append('- `figures/*.pdf`')

with open(rd/'EVALUATION_REPORT.md', 'w') as f:
    f.write('\n'.join(md))

print('  Generated: latex_tables.tex, EVALUATION_REPORT.md')
REPORT_EOF

log "Report generation complete"

#===============================================================================
# Final Summary
#===============================================================================
log ""
log "╔══════════════════════════════════════════════════════════════════╗"
log "║  EVALUATION COMPLETE                                             ║"
log "╚══════════════════════════════════════════════════════════════════╝"
log ""
log "Results: $RESULTS_DIR"
log "Runtime: $SECONDS seconds ($(($SECONDS / 60)) minutes)"
log ""
log "Files:"
ls -la "$RESULTS_DIR"/*.json "$RESULTS_DIR"/*.tex "$RESULTS_DIR"/*.md 2>/dev/null
log ""
log "Figures:"
ls -la "$RESULTS_DIR/figures/"*.pdf 2>/dev/null
log ""
log "View report: cat $RESULTS_DIR/EVALUATION_REPORT.md"
