#!/usr/bin/env bash
# Runs an identical Locust profile against 1, 2, and 4 API replicas and writes
# CSV results per scale. Each replica is capped at 1 CPU (see docker-compose.yml).
set -e
cd "$(dirname "$0")/.."

USERS=50
SPAWN=10
DURATION=60s
RESULTS=deploy/loadtest_results
mkdir -p "$RESULTS"

wait_healthy() {
  for i in $(seq 1 40); do
    code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health --max-time 3 2>/dev/null || true)
    [ "$code" = "200" ] && return 0
    sleep 2
  done
  echo "API did not become healthy" >&2; return 1
}

for N in 1 2 4; do
  echo "=================== SCALE = $N replica(s) ==================="
  docker compose up -d --scale api=$N >/dev/null 2>&1
  # Give freshly started replicas time to load the model before testing.
  sleep $([ "$N" -gt 1 ] && echo 30 || echo 20)
  wait_healthy
  echo "running Locust: $USERS users, ${DURATION}, $N replica(s)..."
  locust -f locustfile.py --host http://localhost:8080 \
         --users $USERS --spawn-rate $SPAWN --run-time $DURATION \
         --headless --only-summary --csv "$RESULTS/scale_$N" >/dev/null 2>&1
  echo "done scale=$N"
done

echo "=================== ALL RUNS COMPLETE ==================="
