#!/usr/bin/env python3
"""
Add documentation to ChromaDB for RAG evaluation.
Expands the corpus from 3 to 20+ documents.
"""

import sys
import os

# SSL fix
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database_handlers import VectorDatabaseHandler

# Technical documentation for common scenarios
DOCUMENTS = [
    # Database troubleshooting
    """# Database Connection Errors

## Symptoms
- Connection refused errors
- Timeout errors after 30s
- Pool exhaustion warnings

## Root Causes
1. Database service not running
2. Network connectivity issues
3. Connection pool saturation
4. Authentication failures

## Solutions
1. Verify database service: `systemctl status mongodb`
2. Check network: `nc -zv localhost 27017`
3. Increase pool size in config
4. Verify credentials in .env file
""",
    
    """# Connection Pool Exhaustion

## Symptoms
- "Connection pool exhausted" errors
- Increasing query latency
- Request timeouts

## Root Cause Analysis
Connection pools have fixed size. Under high load, all connections may be in use.

## Solutions
1. Increase max_pool_size parameter
2. Reduce connection timeout
3. Implement connection retry with backoff
4. Add connection health checks
""",

    # Authentication issues
    """# Authentication Failures

## Symptoms
- 401 Unauthorized responses
- Login failures
- Session expiration errors

## Common Causes
1. Invalid credentials
2. Expired tokens
3. Missing authentication headers
4. CORS blocking requests

## Debugging Steps
1. Check auth logs for specific error codes
2. Verify token expiration times
3. Test with curl: `curl -H "Authorization: Bearer TOKEN" URL`
""",

    """# Rate Limiting Issues

## Symptoms
- 429 Too Many Requests
- "Rate limit exceeded" errors
- Throttled API responses

## Causes
1. Exceeded API quota
2. Too many requests from single IP
3. Bot detection triggered

## Solutions
1. Implement exponential backoff
2. Cache repeated requests
3. Distribute load across IPs
4. Request rate limit increase
""",

    # Memory and resource issues
    """# Memory Leak Debugging

## Symptoms
- Increasing memory usage over time
- OOM (Out of Memory) errors
- Process crashes

## Diagnosis
1. Monitor with `top` or `htop`
2. Use memory profiler: `tracemalloc`
3. Check for circular references

## Solutions
1. Identify and fix memory leaks
2. Increase available memory
3. Implement memory limits
4. Add restart policies
""",

    """# OOM Error Resolution

## Error Message
"MemoryError" or "Killed" (signal 9)

## Immediate Actions
1. Restart the service
2. Check memory consumption patterns
3. Review recent code changes

## Long-term Fixes
1. Optimize data structures
2. Stream large datasets
3. Implement pagination
4. Add memory monitoring alerts
""",

    # Network issues
    """# API Timeout Errors

## Symptoms
- Requests timing out after N seconds
- Gateway timeout (504)
- Connection reset

## Causes
1. Slow upstream service
2. Network congestion
3. Large payload processing
4. Database query bottleneck

## Solutions
1. Increase timeout values
2. Implement async processing
3. Add request caching
4. Optimize slow queries
""",

    """# SSL Certificate Errors

## Error Messages
- "SSL: CERTIFICATE_VERIFY_FAILED"
- "unable to get local issuer certificate"
- "[Errno 2] No such file or directory" for cert

## Causes
1. Missing CA certificates
2. Expired certificates
3. Environment variable pointing to wrong path

## Solutions
1. Install ca-certificates package
2. Unset SSL_CERT_FILE: `unset SSL_CERT_FILE`
3. Update certifi: `pip install --upgrade certifi`
""",

    # Docker and deployment
    """# Docker Container Issues

## Common Problems
1. Container won't start
2. Port already in use
3. Volume permission denied
4. Network connectivity

## Debugging Commands
```bash
docker logs container_name
docker inspect container_name
docker exec -it container_name /bin/bash
docker network ls
```

## Solutions
1. Check logs for startup errors
2. Free the port: `lsof -i :PORT`
3. Fix volume permissions: `chmod 755`
4. Verify network configuration
""",

    """# GPU Not Detected by Ollama

## Symptoms
- Slow inference times
- CPU-only processing
- "No GPU detected" warnings

## Checks
1. Verify GPU driver: `nvidia-smi`
2. Check Docker GPU support: `docker run --gpus all nvidia/cuda nvidia-smi`
3. Verify NVIDIA Container Toolkit

## Solutions
1. Install nvidia-container-toolkit
2. Use `--gpus all` flag
3. Update GPU drivers
4. Restart Docker service
""",

    # Application-specific
    """# Flask Application Startup Failures

## Common Errors
1. "Address already in use"
2. "ImportError: No module named..."
3. "Connection refused" to database

## Debugging Steps
1. Check port availability
2. Verify virtual environment activated
3. Install dependencies: `pip install -r requirements.txt`

## Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
""",

    """# ChromaDB Connection Issues

## Symptoms
- "Connection refused" to port 8000
- Timeout connecting to vector store
- Empty search results

## Causes
1. ChromaDB container not running
2. Wrong host/port configuration
3. Network isolation

## Solutions
1. Start container: `docker start chroma`
2. Verify port mapping: `docker port chroma`
3. Check network: `docker network inspect rca-network`
""",

    """# MongoDB Authentication Errors

## Error Messages
- "Authentication failed"
- "not authorized on database"

## Causes
1. Wrong username/password
2. Database doesn't exist
3. User lacks permissions

## Solutions
1. Verify credentials in .env
2. Create database if needed
3. Grant appropriate roles:
   ```
   use admin
   db.grantRolesToUser("user", ["readWrite"])
   ```
""",

    # Log analysis specific
    """# Log Parsing Best Practices

## Structured Logging
- Use JSON format for machine parsing
- Include timestamp, level, component
- Add trace IDs for distributed systems

## Common Fields
1. timestamp (ISO 8601)
2. level (DEBUG/INFO/WARN/ERROR)
3. message
4. component/module
5. trace_id/request_id

## Anti-patterns
- Unstructured free-text logs
- Missing timestamps
- Inconsistent formats
""",

    """# Root Cause Analysis Methodology

## Steps
1. Identify the earliest error
2. Trace dependencies backward
3. Check for environmental changes
4. Review recent deployments

## Tools
- Log aggregators (ELK, Loki)
- APM solutions (Datadog, New Relic)
- Distributed tracing (Jaeger, Zipkin)

## Common Root Causes
1. Configuration changes
2. Resource exhaustion
3. Dependency failures
4. Code bugs
""",

    # Health checks
    """# Health Check Implementation

## Endpoints
- /health - Basic liveness
- /ready - Readiness (dependencies checked)

## What to Check
1. Database connectivity
2. Cache availability
3. External service reachability
4. Disk space

## Response Format
```json
{
  "status": "healthy",
  "database": "connected",
  "cache": "connected"
}
```
""",

    """# Service Recovery Procedures

## Immediate Response
1. Acknowledge the incident
2. Assess impact scope
3. Implement quick fix

## Recovery Steps
1. Restart affected services
2. Roll back if deployment-related
3. Scale up if resource-related
4. Failover to backup if primary down

## Post-Incident
1. Document timeline
2. Identify root cause
3. Create prevention action items
""",
]

def add_documents_to_chromadb():
    """Add all documents to ChromaDB"""
    print("Initializing VectorDatabaseHandler...")
    handler = VectorDatabaseHandler()
    
    collection = handler.get_collection("docs")
    
    # Get existing count
    existing = collection.count()
    print(f"Existing documents: {existing}")
    
    # Add new documents
    print(f"\nAdding {len(DOCUMENTS)} documents...")
    
    for i, doc in enumerate(DOCUMENTS):
        doc_id = f"doc_{i}_{hash(doc[:50])}"
        try:
            collection.add(
                documents=[doc],
                ids=[doc_id],
            )
            print(f"  Added document {i+1}/{len(DOCUMENTS)}")
        except Exception as e:
            print(f"  Error adding document {i+1}: {e}")
    
    # Verify
    final_count = collection.count()
    print(f"\nFinal document count: {final_count}")
    print(f"Documents added: {final_count - existing}")
    
    return final_count

if __name__ == "__main__":
    count = add_documents_to_chromadb()
    print(f"\n✅ ChromaDB now has {count} documents")
