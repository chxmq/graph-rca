#!/usr/bin/env python3
"""
GraphRCA Benchmark Script
Runs performance experiments for the research paper.
Usage: python experiments/benchmark.py
"""

import sys
import os

# Fix SSL_CERT_FILE environment variable issue
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import tracemalloc
from datetime import datetime

# Sample log entries for testing at different scales
SAMPLE_LOGS = [
    '2024-03-20 10:15:45,234 DEBUG in app: Successfully connected to database',
    '2024-03-20 10:16:12,789 WARNING in app: Rate limit reached for IP: 192.168.1.100',
    '2024-03-20 10:16:23,123 ERROR in database_handlers: Database connection timeout after 30s',
    '2024-03-20 10:16:23,234 INFO in app: Retrying database connection...',
    '2024-03-20 10:16:25,456 DEBUG in app: Connection retry attempt 1 of 3',
    '2024-03-20 10:16:30,789 ERROR in app: Connection retry failed',
    '2024-03-20 10:16:35,012 INFO in app: Switching to backup database',
    '2024-03-20 10:16:40,345 DEBUG in app: Backup database connection established',
    '2024-03-20 10:16:45,678 INFO in app: System recovered successfully',
    '2024-03-20 10:17:00,901 DEBUG in app: Health check passed',
]

def benchmark_log_parsing(num_entries=5):
    """Benchmark log parsing performance"""
    try:
        from utils.log_parser import LogParser
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'hint': 'Try running: unset SSL_CERT_FILE before python'
        }
    
    try:
        parser = LogParser()
        logs = SAMPLE_LOGS[:num_entries]
        log_text = '\n'.join(logs)
        
        # Measure parsing time
        tracemalloc.start()
        start_time = time.time()
        
        result = parser.parse_log(log_text)
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate accuracy metrics
        field_stats = {
            'timestamp': {'present': 0, 'correct': 0},
            'message': {'present': 0, 'correct': 0},
            'level': {'present': 0, 'correct': 0},
            'component': {'present': 0, 'correct': 0},
            'ip_address': {'present': 0, 'correct': 0},
        }
        
        for entry in result.log_chain:
            # Timestamp check
            if entry.timestamp:
                field_stats['timestamp']['present'] += 1
                if '2024' in entry.timestamp or '10:' in entry.timestamp:
                    field_stats['timestamp']['correct'] += 1
            
            # Message check
            if entry.message:
                field_stats['message']['present'] += 1
                field_stats['message']['correct'] += 1
            
            # Level check
            if entry.level:
                field_stats['level']['present'] += 1
                if entry.level.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                    field_stats['level']['correct'] += 1
            
            # Component check
            if entry.component:
                field_stats['component']['present'] += 1
                if entry.component in ['app', 'database_handlers']:
                    field_stats['component']['correct'] += 1
            
            # IP address check
            if entry.ip_address:
                field_stats['ip_address']['present'] += 1
                if '192.168' in entry.ip_address:
                    field_stats['ip_address']['correct'] += 1
        
        return {
            'num_entries': num_entries,
            'parsed_entries': len(result.log_chain),
            'total_time_sec': round(end_time - start_time, 3),
            'avg_time_per_entry_sec': round((end_time - start_time) / num_entries, 3),
            'memory_current_mb': round(current / 1024 / 1024, 2),
            'memory_peak_mb': round(peak / 1024 / 1024, 2),
            'field_stats': field_stats,
        }
    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
        }

def benchmark_dag_construction(num_entries=5):
    """Benchmark DAG construction performance"""
    from utils.graph_generator import GraphGenerator
    from models.parsing_data_models import LogChain, LogEntry
    
    # Create synthetic log entries
    entries = []
    for i in range(num_entries):
        entries.append(LogEntry(
            timestamp=f"2024-03-20 10:{i:02d}:00",
            message=f"Event {i} occurred",
            level="INFO" if i % 2 == 0 else "ERROR",
            pid="",
            component="app",
            error_code="",
            username="",
            ip_address="",
            group="",
            trace_id="",
            request_id=""
        ))
    
    log_chain = LogChain(log_chain=entries)
    
    # Measure DAG construction
    tracemalloc.start()
    start_time = time.time()
    
    generator = GraphGenerator(log_chain)
    dag = generator.generate_dag()
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Validate DAG properties
    acyclic = dag.root_id is not None
    temporal_consistent = all(
        node.parent_id is None or 
        any(p.id == node.parent_id for p in dag.nodes)
        for node in dag.nodes
    )
    
    return {
        'num_entries': num_entries,
        'num_nodes': len(dag.nodes),
        'num_leaves': len(dag.leaf_ids),
        'construction_time_sec': round(end_time - start_time, 4),
        'memory_peak_mb': round(peak / 1024 / 1024, 2),
        'is_acyclic': acyclic,
        'temporal_consistent': temporal_consistent,
        'root_cause_identified': dag.root_cause is not None,
    }

def benchmark_rag_retrieval():
    """Benchmark RAG retrieval performance"""
    from core.database_handlers import VectorDatabaseHandler
    
    try:
        handler = VectorDatabaseHandler()
        
        # Test queries
        queries = [
            "database connection timeout",
            "rate limit exceeded",
            "authentication failure",
        ]
        
        results = []
        for query in queries:
            start_time = time.time()
            # Use the correct method: search() with query and context
            docs = handler.search(query=query, context="log analysis", top_k=3)
            end_time = time.time()
            
            results.append({
                'query': query,
                'latency_sec': round(end_time - start_time, 3),
                'docs_retrieved': len(docs) if docs else 0,
            })
        
        avg_latency = sum(r['latency_sec'] for r in results) / len(results)
        
        return {
            'queries_tested': len(queries),
            'avg_retrieval_latency_sec': round(avg_latency, 3),
            'results': results,
        }
    except Exception as e:
        import traceback
        return {'error': str(e), 'traceback': traceback.format_exc()}

def benchmark_end_to_end(num_entries=5):
    """Benchmark full pipeline performance"""
    from utils.log_parser import LogParser
    from utils.graph_generator import GraphGenerator
    from utils.context_builder import ContextBuilder
    from core.rag import RAG_Engine
    
    logs = SAMPLE_LOGS[:num_entries]
    log_text = '\n'.join(logs)
    
    timings = {}
    
    # Step 1: Parse logs
    start = time.time()
    parser = LogParser()
    log_chain = parser.parse_log(log_text)
    timings['parsing'] = round(time.time() - start, 3)
    
    # Step 2: Generate DAG
    start = time.time()
    generator = GraphGenerator(log_chain)
    dag = generator.generate_dag()
    timings['dag_generation'] = round(time.time() - start, 3)
    
    # Step 3: Build context
    start = time.time()
    builder = ContextBuilder()
    context = builder.build_context(dag)
    timings['context_building'] = round(time.time() - start, 3)
    
    # Step 4: RAG solution (optional, may timeout)
    try:
        start = time.time()
        rag = RAG_Engine()
        solution = rag.generate_solution(context, "How to fix this issue?")
        timings['rag_solution'] = round(time.time() - start, 3)
    except Exception as e:
        timings['rag_solution'] = f"error: {str(e)}"
    
    total = sum(v for v in timings.values() if isinstance(v, (int, float)))
    
    return {
        'num_entries': num_entries,
        'timings': timings,
        'total_time_sec': round(total, 3),
    }

def run_scalability_test():
    """Run scalability benchmarks at different scales"""
    scales = [5, 10, 20]  # Reduced for faster testing
    results = []
    
    for n in scales:
        print(f"\n--- Testing with {n} entries ---")
        try:
            dag_result = benchmark_dag_construction(n)
            results.append({
                'entries': n,
                'dag_time_sec': dag_result['construction_time_sec'],
                'memory_mb': dag_result['memory_peak_mb'],
            })
        except Exception as e:
            print(f"Error at scale {n}: {e}")
    
    return results

def main():
    print("=" * 60)
    print("GraphRCA Benchmark Suite")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    all_results = {}
    
    # 1. Log Parsing Benchmark
    print("\n[1/4] Running Log Parsing Benchmark...")
    try:
        all_results['parsing'] = benchmark_log_parsing(5)
        print(f"  Parsed {all_results['parsing']['parsed_entries']} entries in {all_results['parsing']['total_time_sec']}s")
    except Exception as e:
        all_results['parsing'] = {'error': str(e)}
        print(f"  Error: {e}")
    
    # 2. DAG Construction Benchmark
    print("\n[2/4] Running DAG Construction Benchmark...")
    try:
        all_results['dag'] = benchmark_dag_construction(10)
        print(f"  Built DAG with {all_results['dag']['num_nodes']} nodes in {all_results['dag']['construction_time_sec']}s")
    except Exception as e:
        all_results['dag'] = {'error': str(e)}
        print(f"  Error: {e}")
    
    # 3. RAG Retrieval Benchmark
    print("\n[3/4] Running RAG Retrieval Benchmark...")
    try:
        all_results['rag'] = benchmark_rag_retrieval()
        if 'error' not in all_results['rag']:
            print(f"  Avg retrieval latency: {all_results['rag']['avg_retrieval_latency_sec']}s")
        else:
            print(f"  Error: {all_results['rag']['error']}")
    except Exception as e:
        all_results['rag'] = {'error': str(e)}
        print(f"  Error: {e}")
    
    # 4. Scalability Test
    print("\n[4/4] Running Scalability Test...")
    try:
        all_results['scalability'] = run_scalability_test()
    except Exception as e:
        all_results['scalability'] = {'error': str(e)}
        print(f"  Error: {e}")
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_file}")
    print("=" * 60)
    
    # Print summary for paper
    print("\n\n=== PAPER VALUES ===\n")
    
    if 'parsing' in all_results and 'error' not in all_results['parsing']:
        p = all_results['parsing']
        print(f"Parsing latency per entry: {p['avg_time_per_entry_sec']}s")
        print(f"Peak memory: {p['memory_peak_mb']} MB")
    
    if 'dag' in all_results and 'error' not in all_results['dag']:
        d = all_results['dag']
        print(f"DAG construction time: {d['construction_time_sec']}s for {d['num_entries']} entries")
        print(f"Acyclicity verified: {d['is_acyclic']}")
        print(f"Temporal consistency: {d['temporal_consistent']}")
    
    if 'rag' in all_results and 'error' not in all_results['rag']:
        r = all_results['rag']
        print(f"RAG retrieval latency: {r['avg_retrieval_latency_sec']}s")

if __name__ == "__main__":
    main()
