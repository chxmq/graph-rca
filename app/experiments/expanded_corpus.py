#!/usr/bin/env python3
"""
Expanded documentation corpus for rigorous RAG evaluation.
50+ documents covering all major incident categories.
"""

import sys
import os

# SSL fix
if 'SSL_CERT_FILE' in os.environ:
    del os.environ['SSL_CERT_FILE']

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# DATABASE TROUBLESHOOTING (10 documents)
# ============================================================
DATABASE_DOCS = [
    """# MongoDB Connection Timeout
## Symptoms
- Connection timeout after 30 seconds
- "ServerSelectionTimeoutError" messages
- Queries hanging indefinitely

## Root Causes
1. MongoDB service not running
2. Network firewall blocking port 27017
3. Incorrect connection string
4. Too many concurrent connections

## Solutions
1. Check MongoDB status: `sudo systemctl status mongod`
2. Verify port access: `nc -zv localhost 27017`
3. Review connection string in .env file
4. Increase maxPoolSize or close idle connections""",

    """# PostgreSQL Connection Pool Exhaustion
## Symptoms
- "connection pool exhausted" errors
- Increasing query latency over time
- Application hangs during high load

## Diagnosis
1. Check active connections: `SELECT count(*) FROM pg_stat_activity`
2. Identify long-running queries
3. Review connection pool configuration

## Solutions
1. Increase max_connections in postgresql.conf
2. Implement connection timeout
3. Use connection pooler (PgBouncer)
4. Add connection leak detection""",

    """# MySQL Deadlock Detection
## Symptoms
- "Deadlock found" error messages
- Transactions rolling back unexpectedly
- Intermittent write failures

## Diagnosis
1. Check InnoDB status: `SHOW ENGINE INNODB STATUS`
2. Review deadlock log section
3. Identify conflicting transactions

## Solutions
1. Reorder operations for consistent lock ordering
2. Keep transactions short
3. Use row-level locking where possible
4. Implement retry logic with backoff""",

    """# Redis Memory Overflow
## Symptoms
- "OOM command not allowed" errors
- Write operations failing
- Eviction warnings in logs

## Diagnosis
1. Check memory: `redis-cli INFO memory`
2. Review eviction policy
3. Identify large keys: `redis-cli --bigkeys`

## Solutions
1. Increase maxmemory setting
2. Configure appropriate eviction policy
3. Implement TTL for all keys
4. Consider Redis Cluster for scaling""",

    """# Elasticsearch Cluster Yellow/Red Status
## Symptoms
- Cluster health yellow or red
- Search operations timing out
- Unassigned shards

## Diagnosis
1. Check cluster health: `GET /_cluster/health`
2. List unassigned shards: `GET /_cat/shards?h=index,shard,state`
3. Review allocation explain

## Solutions
1. Add more nodes for replica allocation
2. Reduce replica count temporarily
3. Check disk space thresholds
4. Force shard reallocation if corrupted""",

    """# SQLite Database Lock
## Symptoms
- "database is locked" errors
- Concurrent write failures
- Timeout during transactions

## Root Causes
1. Long-running write transactions
2. Multiple processes accessing database
3. WAL mode not enabled

## Solutions
1. Enable WAL mode: `PRAGMA journal_mode=WAL`
2. Use connection pooling
3. Implement write queue
4. Consider alternative database for high concurrency""",

    """# Cassandra Node Failure
## Symptoms
- "NoHostAvailableException" errors
- Inconsistent read results
- Timeouts during writes

## Diagnosis
1. Check nodetool status
2. Review system.log for errors
3. Verify gossip communication

## Solutions
1. Restart failed node
2. Check network connectivity
3. Repair inconsistent data: `nodetool repair`
4. Bootstrap replacement node if hardware failed""",

    """# Database Replication Lag
## Symptoms
- Stale reads from replicas
- Increasing lag metrics
- Eventual consistency violations

## Diagnosis
1. Check replication status
2. Monitor lag in seconds
3. Review network latency

## Solutions
1. Optimize primary write load
2. Upgrade replica hardware
3. Reduce network latency
4. Use synchronous replication for critical data""",

    """# Index Bloat and Performance
## Symptoms
- Queries slowing over time
- Index size growing unexpectedly
- High I/O during maintenance

## Diagnosis
1. Check index size vs table size
2. Identify unused indexes
3. Review bloat estimation queries

## Solutions
1. REINDEX to rebuild bloated indexes
2. Drop unused indexes
3. Schedule regular VACUUM (PostgreSQL)
4. Implement partitioning for large tables""",

    """# Database Backup Failure
## Symptoms
- Backup job failed notification
- Incomplete backup files
- Timeout during backup

## Root Causes
1. Insufficient disk space
2. Database lock during backup
3. Network interruption
4. Corrupt source data

## Solutions
1. Verify backup destination space
2. Use hot backup methods
3. Implement backup monitoring
4. Test backup restoration regularly""",
]

# ============================================================
# AUTHENTICATION & SECURITY (10 documents)
# ============================================================  
AUTH_DOCS = [
    """# JWT Token Expiration
## Symptoms
- 401 Unauthorized after period of inactivity
- "Token expired" error messages
- Users forced to re-login frequently

## Solutions
1. Implement refresh token mechanism
2. Adjust token expiration time
3. Add silent token refresh on frontend
4. Cache user sessions server-side""",

    """# OAuth2 Callback Failure
## Symptoms
- "redirect_uri mismatch" errors
- OAuth flow not completing
- Stuck on provider authorization page

## Root Causes
1. Mismatched redirect URI in app registration
2. HTTPS vs HTTP mismatch
3. Port number difference

## Solutions
1. Verify redirect URI in OAuth provider settings
2. Ensure exact match including protocol
3. Update callback URL in application config""",

    """# CORS Preflight Failure
## Symptoms
- "Access-Control-Allow-Origin" errors
- OPTIONS requests failing
- Cross-origin API calls blocked

## Solutions
1. Configure CORS middleware properly
2. Add allowed origins to configuration
3. Handle OPTIONS preflight explicitly
4. Verify headers in response""",

    """# Session Fixation Attack
## Symptoms
- Security audit flagged session handling
- Session IDs not rotating on login
- Potential account hijacking

## Solutions
1. Regenerate session ID on authentication
2. Implement secure session cookie flags
3. Add session timeout mechanism
4. Log session creation events""",

    """# Brute Force Detection
## Symptoms
- Multiple failed login attempts
- Account lockouts
- Suspicious IP addresses

## Solutions
1. Implement rate limiting per IP
2. Add CAPTCHA after failed attempts
3. Use exponential backoff
4. Alert on multiple failures""",

    """# API Key Compromise
## Symptoms
- Unexpected API usage spikes
- Unauthorized operations in logs
- API key exposed in repository

## Immediate Actions
1. Revoke compromised key immediately
2. Generate new API key
3. Rotate all related secrets
4. Audit access logs

## Prevention
1. Never commit secrets to git
2. Use environment variables
3. Implement key rotation policy""",

    """# SSL Certificate Errors
## Symptoms
- Browser security warnings
- "certificate expired" errors
- HTTPS connections failing

## Solutions
1. Renew SSL certificate
2. Update certificate chain
3. Configure auto-renewal (Let's Encrypt)
4. Verify certificate installation""",

    """# Password Hash Weakness
## Symptoms
- Security audit finding
- Using deprecated hash algorithm
- Fast brute-force possibility

## Solutions
1. Migrate to bcrypt/argon2
2. Increase work factor
3. Force password reset for users
4. Implement password policy""",

    """# Two-Factor Authentication Bypass
## Symptoms
- 2FA not being enforced
- Backup codes not working
- TOTP time sync issues

## Solutions
1. Enforce 2FA on all login paths
2. Sync server time (NTP)
3. Implement backup recovery flow
4. Log all 2FA attempts""",

    """# Role Permission Escalation
## Symptoms
- Users accessing unauthorized resources
- Permission checks bypassed
- Admin functions exposed

## Solutions
1. Implement defense in depth
2. Check permissions at every layer
3. Use principle of least privilege
4. Audit permission changes""",
]

# ============================================================
# MEMORY & RESOURCE ISSUES (10 documents)
# ============================================================
MEMORY_DOCS = [
    """# Java Heap Space Error
## Symptoms
- "java.lang.OutOfMemoryError: Java heap space"
- Application crash after running for hours
- Increasing memory usage over time

## Diagnosis
1. Analyze heap dump: `jmap -dump:format=b,file=heap.bin <pid>`
2. Use MAT or VisualVM
3. Identify memory-holding objects

## Solutions
1. Increase -Xmx heap size
2. Fix memory leaks in code
3. Implement object pooling
4. Use WeakReferences where appropriate""",

    """# Linux OOM Killer
## Symptoms
- Process killed by signal 9
- "Out of memory: Kill process" in dmesg
- Application disappearing without logs

## Diagnosis
1. Check dmesg for OOM messages
2. Review /proc/meminfo
3. Identify memory hog processes

## Solutions
1. Increase system memory
2. Adjust OOM score (-1000 to protect)
3. Enable swap space
4. Optimize application memory usage""",

    """# Container Memory Limits
## Symptoms
- Container OOMKilled
- Restart loop in Kubernetes
- Memory limit exceeded

## Solutions
1. Increase container memory limits
2. Optimize application memory
3. Tune JVM for container awareness
4. Use memory requests appropriately""",

    """# Memory Leak Detection
## Symptoms
- Gradual memory increase over time
- Performance degradation
- Eventual crash or OOM

## Diagnosis (Python)
1. Use tracemalloc module
2. Take memory snapshots
3. Compare allocations over time

## Solutions
1. Fix circular references
2. Clear caches periodically
3. Use __slots__ for memory efficiency
4. Profile with memory_profiler""",

    """# CPU Throttling
## Symptoms
- Inconsistent response times
- CPU capped at certain percentage
- Kubernetes CPU throttling metrics

## Solutions
1. Increase CPU limits
2. Optimize hot code paths
3. Implement caching
4. Scale horizontally""",

    """# Disk Space Exhaustion
## Symptoms
- "No space left on device" errors
- Write operations failing
- Services refusing to start

## Immediate Actions
1. Identify large files: `du -sh /*`
2. Clear log files: `truncate -s 0 /var/log/*.log`
3. Remove old Docker images
4. Clear package cache

## Prevention
1. Implement log rotation
2. Monitor disk usage
3. Set up alerts at 80%""",

    """# File Descriptor Exhaustion
## Symptoms
- "Too many open files" errors
- Cannot open new connections
- Services failing to accept requests

## Diagnosis
1. Check limits: `ulimit -n`
2. Count open FDs: `ls /proc/<pid>/fd | wc -l`
3. Identify leaking descriptors

## Solutions
1. Increase ulimit in systemd service
2. Fix connection/file leaks
3. Configure soft/hard limits in limits.conf""",

    """# Thread Pool Exhaustion
## Symptoms
- Requests queuing
- Timeout errors increasing
- Thread count at maximum

## Solutions
1. Increase thread pool size
2. Implement async operations
3. Add request timeout
4. Use connection pooling""",

    """# Garbage Collection Pause
## Symptoms
- Periodic latency spikes
- Application freezing briefly
- GC logs showing long pauses

## Solutions
1. Tune GC algorithm (G1, ZGC)
2. Reduce heap size
3. Minimize object allocation
4. Profile allocation patterns""",

    """# Cache Memory Overflow
## Symptoms
- Cache eviction warnings
- Miss rate increasing
- Memory pressure alerts

## Solutions
1. Implement LRU eviction
2. Set maximum cache size
3. Add TTL for entries
4. Monitor cache hit ratio""",
]

# ============================================================
# NETWORK & CONNECTIVITY (10 documents)
# ============================================================
NETWORK_DOCS = [
    """# DNS Resolution Failure
## Symptoms
- "Name or service not known" errors
- Intermittent connection failures
- Slow service discovery

## Solutions
1. Check /etc/resolv.conf
2. Verify DNS server availability
3. Use explicit IP for critical services
4. Implement DNS caching""",

    """# Load Balancer Health Check Failure
## Symptoms
- Instances marked unhealthy
- Traffic not reaching servers
- 502/503 gateway errors

## Solutions
1. Verify health check endpoint
2. Adjust timeout settings
3. Fix application startup time
4. Review health check criteria""",

    """# TCP Connection Reset
## Symptoms
- "Connection reset by peer" errors
- Frequent reconnections
- Data transfer interruptions

## Root Causes
1. Firewall timeout
2. Server-side connection limit
3. Network equipment issues
4. Application crash

## Solutions
1. Implement connection keepalive
2. Handle reconnection gracefully
3. Increase timeout values""",

    """# Proxy Timeout
## Symptoms
- 504 Gateway Timeout
- Long-running requests failing
- Nginx proxy_read_timeout errors

## Solutions
1. Increase proxy timeout settings
2. Optimize slow backend operations
3. Implement async processing
4. Add progress feedback""",

    """# Service Mesh Connectivity
## Symptoms
- Istio sidecar injection issues
- mTLS handshake failures
- Service-to-service communication blocked

## Solutions
1. Verify sidecar injection
2. Check mTLS configuration
3. Review network policies
4. Debug with istioctl analyze""",

    """# Kubernetes Network Policy
## Symptoms
- Pods cannot communicate
- Ingress/egress blocked
- DNS not resolving

## Solutions
1. Review NetworkPolicy rules
2. Allow required namespaces
3. Permit DNS egress (port 53)
4. Debug with network tools pod""",

    """# WebSocket Connection Failure
## Symptoms
- WebSocket handshake failing
- Connection dropping immediately
- Proxy not supporting upgrade

## Solutions
1. Configure proxy for WebSocket
2. Enable Connection: Upgrade header
3. Increase timeout for long-lived connections
4. Implement reconnection logic""",

    """# API Rate Limiting
## Symptoms
- 429 Too Many Requests
- Throttled API responses
- Request queuing

## Solutions
1. Implement exponential backoff
2. Add request caching
3. Batch API calls
4. Request rate limit increase""",

    """# SSL/TLS Handshake Failure
## Symptoms
- "SSL handshake failed" errors
- Certificate verification failing
- Protocol version mismatch

## Solutions
1. Update SSL/TLS library
2. Use compatible protocol version
3. Verify certificate chain
4. Check hostname matching""",

    """# gRPC Connectivity Issues
## Symptoms
- gRPC deadline exceeded
- Connection refused
- Load balancing not working

## Solutions
1. Verify gRPC health checks
2. Configure client-side LB
3. Enable keepalive pings
4. Review channel settings""",
]

# ============================================================
# DEPLOYMENT & CONTAINERIZATION (10 documents)
# ============================================================
DEPLOYMENT_DOCS = [
    """# Docker Build Failure
## Symptoms
- "docker build" command failing
- Layer caching not working
- Dependency resolution errors

## Solutions
1. Check Dockerfile syntax
2. Verify base image availability
3. Clear build cache: `docker builder prune`
4. Multi-stage build for smaller images""",

    """# Container Startup Crash
## Symptoms
- Container exits immediately
- Restart loop detected
- No logs in container

## Diagnosis
1. Check exit code: `docker inspect`
2. View logs: `docker logs <container>`
3. Run interactively: `docker run -it <image> /bin/sh`

## Solutions
1. Fix entrypoint command
2. Ensure dependencies available
3. Check file permissions""",

    """# Kubernetes Pod CrashLoopBackOff
## Symptoms
- Pod continuously restarting
- CrashLoopBackOff status
- Container failing health checks

## Diagnosis
1. Describe pod: `kubectl describe pod`
2. View logs: `kubectl logs --previous`
3. Check events

## Solutions
1. Fix application startup errors
2. Adjust resource limits
3. Review liveness probe settings""",

    """# Image Pull Failure
## Symptoms
- ErrImagePull or ImagePullBackOff
- Authentication errors
- Image not found

## Solutions
1. Verify image name and tag
2. Create imagePullSecrets
3. Check registry connectivity
4. Login to private registry""",

    """# Helm Chart Deployment Failure
## Symptoms
- helm install/upgrade failing  
- Template rendering errors
- Resource conflicts

## Solutions
1. Validate chart: `helm lint`
2. Debug templates: `helm template --debug`
3. Check values file syntax
4. Review Kubernetes API versions""",

    """# Rolling Update Stuck
## Symptoms
- Deployment not progressing
- Old pods not terminating
- New pods failing readiness

## Solutions
1. Check readiness probe
2. Verify resource availability
3. Review pod disruption budget
4. Rollback if necessary: `kubectl rollout undo`""",

    """# ConfigMap/Secret Not Updating
## Symptoms
- Application using old config
- Changes not reflected
- Pod not restarting on update

## Solutions
1. Trigger pod restart
2. Use checksum annotation
3. Implement config reloader
4. Mount as volume (not env)""",

    """# Volume Mount Failure
## Symptoms
- Pod stuck in ContainerCreating
- "Unable to mount volumes" error
- Storage class issues

## Solutions
1. Verify PVC is bound
2. Check storage class exists
3. Review node selectors
4. Check storage driver logs""",

    """# Ingress Not Routing
## Symptoms
- 404 from ingress
- Backend not receiving traffic
- TLS termination failing

## Solutions
1. Verify ingress class annotation
2. Check service selectors
3. Review ingress controller logs
4. Validate TLS secret""",

    """# Resource Quota Exceeded
## Symptoms
- Pods pending creation
- "exceeded quota" errors
- Namespace limits reached

## Solutions
1. Delete unused resources
2. Request quota increase
3. Optimize resource requests
4. Use autoscaling""",
]

# Combine all documents
ALL_DOCUMENTS = DATABASE_DOCS + AUTH_DOCS + MEMORY_DOCS + NETWORK_DOCS + DEPLOYMENT_DOCS

def add_all_documents():
    """Add all 50 documents to ChromaDB."""
    from core.database_handlers import VectorDatabaseHandler
    
    print("="*60)
    print("EXPANDED CORPUS LOADER")
    print(f"Total documents: {len(ALL_DOCUMENTS)}")
    print("="*60)
    
    handler = VectorDatabaseHandler()
    collection = handler.get_collection("docs")
    
    existing = collection.count()
    print(f"Existing documents: {existing}")
    
    # Add in batches
    added = 0
    failed = 0
    
    for i, doc in enumerate(ALL_DOCUMENTS):
        doc_id = f"corpus_{i}_{hash(doc[:30]) % 10000}"
        try:
            collection.add(
                documents=[doc],
                ids=[doc_id],
                metadatas=[{"category": get_category(i), "index": i}]
            )
            added += 1
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(ALL_DOCUMENTS)}")
        except Exception as e:
            failed += 1
            if "already exists" not in str(e).lower():
                print(f"  Error on doc {i}: {e}")
    
    final = collection.count()
    print(f"\n✅ Added: {added}, Failed: {failed}")
    print(f"Total documents in corpus: {final}")
    
    return final

def get_category(index: int) -> str:
    """Get category based on document index."""
    if index < 10:
        return "database"
    elif index < 20:
        return "authentication"
    elif index < 30:
        return "memory"
    elif index < 40:
        return "network"
    else:
        return "deployment"

if __name__ == "__main__":
    count = add_all_documents()
    print(f"\nCorpus now contains {count} documents")
