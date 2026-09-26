#!/bin/bash
# one line per job: done (exit, seconds) or running (last log line)
ROOT=${BOX_ROOT:-/root/job}
now=$(date +%s); start=$(cat "$ROOT/logs/START" 2>/dev/null || echo "$now")
echo "elapsed $(( (now - start) / 60 )) min; load $(cut -d' ' -f1-3 /proc/loadavg)"
for f in "$ROOT"/logs/*.log; do
  n=$(basename "$f" .log)
  if [ -f "$ROOT/logs/$n.done" ]; then
    echo "  $n: DONE $(cat "$ROOT/logs/$n.done")"
  else
    echo "  $n: running | $(grep -v '^\s*$' "$f" | tail -n 1 | cut -c1-110)"
  fi
done
[ -f "$ROOT/logs/ALL.done" ] && echo "ALL DONE"
grep -l "Traceback" "$ROOT"/logs/*.log 2>/dev/null | sed 's/^/  TRACEBACK in /'
