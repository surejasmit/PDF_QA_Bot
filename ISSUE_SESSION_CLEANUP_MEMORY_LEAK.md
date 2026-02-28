# 🔴 Critical Issue: FastAPI Session Cleanup Never Runs Automatically

**Severity**: High  
**Type**: Memory Leak / Resource Exhaustion  
**Component**: `rag-service/main.py`  
**Status**: Open

---

## Problem Summary

The FastAPI service has a **reactive cleanup mechanism** that only runs when new requests arrive. If the service becomes idle (no incoming requests), expired sessions accumulate indefinitely in memory, causing a **memory leak**.

---

## Technical Details

### Current Implementation

**Location**: `rag-service/main.py` lines 180-185

```python
def cleanup_sessions():
    now = time.time()
    expired = [k for k, v in sessions.items()
               if now - v["last"] > SESSION_TIMEOUT]
    for k in expired:
        del sessions[k]
```

**Called in**:
- `/process` endpoint (line 192)
- `/ask` endpoint (line 219)

### The Problem

1. **Reactive, Not Proactive**: `cleanup_sessions()` only executes when `/process` or `/ask` endpoints receive requests
2. **Idle Service Scenario**: If no requests arrive for hours, expired sessions remain in memory
3. **Memory Growth**: Each session stores a FAISS vector store (can be 10-100+ MB per document)
4. **No Automatic Cleanup**: Python garbage collector won't clean up because sessions dict still references them

---

## Impact Analysis

### Memory Consumption Example

```
User uploads 50 PDFs over 2 hours → 50 sessions created
Service becomes idle for 3 hours (SESSION_TIMEOUT = 1 hour)
Expected: 0 sessions in memory (all expired)
Actual: 50 sessions still in memory (5+ GB wasted)
```

### Real-World Scenarios

- **Low-traffic periods**: Nights/weekends with no cleanup
- **Burst traffic followed by silence**: All sessions persist indefinitely
- **Container/VM memory limits**: Service crashes with OOM errors
- **Cloud costs**: Unnecessary memory usage increases hosting costs

---

## Root Cause

The design assumes **continuous traffic** to trigger cleanup. This violates the principle of **autonomous resource management**.

---

## Reproduction Steps

1. Start FastAPI service: `uvicorn main:app --port 5000`
2. Upload 10 PDFs via `/process` endpoint
3. Wait for `SESSION_TIMEOUT + 1 hour` (default: 2 hours total)
4. Check memory usage: `ps aux | grep uvicorn`
5. **Result**: Memory still high, sessions dict still contains 10 entries

---

## Proposed Solutions

### Solution 1: Background Cleanup Task (Recommended)

Add a periodic background task using FastAPI's lifespan events:

```python
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cleanup_task = asyncio.create_task(periodic_cleanup())
    yield
    # Shutdown
    cleanup_task.cancel()

async def periodic_cleanup():
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        cleanup_sessions()

app = FastAPI(lifespan=lifespan)
```

### Solution 2: TTL-based Session Store

Use Redis or similar with built-in expiration:

```python
import redis
r = redis.Redis(host='localhost', port=6379)
r.setex(f"session:{session_id}", SESSION_TIMEOUT, pickle.dumps(data))
```

---

## Recommended Fix

Implement **Solution 1** (minimal code change, no external dependencies).

**Estimated effort**: 15 minutes  
**Risk**: Low (non-breaking change)

---

## Related Issues

- Similar to disk space exhaustion (#110) but for memory
- Affects scalability and production stability
- Not covered in existing GitHub issues

---

## Testing Verification

After fix, verify:
1. Memory usage decreases after `SESSION_TIMEOUT` even without new requests
2. `len(sessions)` returns 0 after idle period
3. Service remains stable under low-traffic conditions
