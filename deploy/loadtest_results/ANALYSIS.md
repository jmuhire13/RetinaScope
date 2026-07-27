# Load Test Results — Horizontal Scaling

**Tool:** Locust · **Endpoint under load:** `POST /predict` (real retina images) ·
**Profile (identical across all runs):** 50 concurrent users, spawn rate 10/s,
60-second run, through the nginx load balancer at `http://localhost:8080`.

**Setup:** local `docker-compose` with nginx round-robin load balancing across
1, 2, then 4 API replicas. Each replica is capped at **1 CPU** so that adding
replicas adds real serving capacity (without the cap, one replica's TensorFlow
intra-op threading would consume all host cores and hide any scaling effect).
Run on an 8-core host, leaving ≥4 cores for the Locust load generator + nginx.

> The container-count comparison is run **locally**, not on the Render free tier,
> which does not permit replica scaling. Render hosts the public single-instance
> prediction URL; this local harness is the scaling demonstration. Two separate,
> both-legitimate demonstrations.

## Results

| Replicas | Requests (60s) | Failures | Throughput (req/s) | Median latency | p95 latency | Max latency |
|:--------:|:--------------:|:--------:|:------------------:|:--------------:|:-----------:|:-----------:|
| 1 | 90  | 0 | 1.5  | 27.0 s | 32.0 s | 33.6 s |
| 2 | 202 | 0 | 3.4  | 11.0 s | 19.0 s | 21.8 s |
| 4 | 595 | 0 | 10.1 | 3.9 s  | 9.5 s  | 13.4 s |

![Scaling chart](scaling_chart.png)

## What the numbers say

- **Throughput scales strongly with replicas:** 1.5 → 3.4 → 10.1 req/s
  (2.3× at 2 replicas, 6.7× at 4 replicas vs. the single replica).
- **Latency collapses:** median 27.0 s → 3.9 s, p95 32.0 s → 9.5 s as replicas go 1 → 4.
- **Zero failures at every scale** — the system degrades gracefully (by queuing,
  not erroring) even when a single replica is heavily overloaded.

## Honest interpretation

The scaling is **super-linear** (6.7× throughput for 4× the CPU), which is not a
free lunch — it reflects the *regime* the single replica is in. With 50 concurrent
users hitting one 1-CPU replica that serves CPU-bound CNN inference essentially one
request at a time, the single-replica case is in **deep saturation**: requests queue
for ~27 s, and few complete inside the 60 s window, which depresses its measured
throughput. Adding replicas relieves that queue disproportionately until the system
approaches an unsaturated regime — hence the greater-than-linear gains. The correct
takeaway is therefore qualitative and strong: **horizontal scaling is essential for
this workload under load, and each replica added yields a large, measurable
improvement in both throughput and tail latency.**

The absolute latencies are high by design — this is a deliberate *flood* (50 users,
CPU-only inference, 1 CPU per replica) meant to stress the system, not a
representative single-user latency (a single prediction served in isolation returns
in well under a second). The value of the test is the **relative** behaviour across
container counts.

**Caveat:** the Locust generator runs on the same host as the containers. Cores were
reserved for it (≤4 replica CPUs on an 8-core host), and the clean upward throughput
trend confirms the generator was not itself the bottleneck at these levels.
