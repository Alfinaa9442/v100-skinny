#!/usr/bin/env bash
# Serving lifecycle: PID-file based, graceful-first. Usage:
#   serve_ctl.sh stop            — TERM the recorded server group, wait
#   serve_ctl.sh wait            — wait for /v1/models health
set -u
PIDFILE="${SERVE_PIDFILE:-$HOME/1cat-122/serve.pid}"
case "${1:-}" in
  stop)
    if [ -f "$PIDFILE" ]; then
      PID=$(cat "$PIDFILE")
      PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d " ")
      if [ -n "${PGID:-}" ]; then
        kill -TERM -- "-$PGID" 2>/dev/null
        for i in $(seq 1 30); do
          kill -0 "$PID" 2>/dev/null || { echo stopped; rm -f "$PIDFILE"; exit 0; }
          sleep 2
        done
        kill -KILL -- "-$PGID" 2>/dev/null
        echo "stopped (escalated after 60s)"
      else
        echo "no such process"
      fi
      rm -f "$PIDFILE"
    else
      echo "no pidfile"
    fi
    ;;
  wait)
    for i in $(seq 1 360); do
      curl -s --max-time 2 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen3.6 && { echo up; exit 0; }
      sleep 5
    done
    echo timeout; exit 1
    ;;
esac
