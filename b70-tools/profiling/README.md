# Profiling a warm server with unitrace

`build-unitrace.sh` builds Intel's `unitrace`. To trace only the part that matters (not load, not kernel JIT),
start the server paused, warm it up, then trace one request:

```sh
unitrace -d -h --start-paused --session qwfn --teardown-on-signal 15 \
  build/qwfn-server HEAD --host 127.0.0.1 --port 8096 ... > server.log 2>&1 &
python3 tools/perf/decab.py 8096 warm 2 0 256       # warm-up, untraced
unitrace --resume qwfn
python3 tools/perf/decab.py 8096 traced 0 1 256     # the traced request
unitrace --pause qwfn
kill -TERM $(pgrep -x qwfn-server)                   # --teardown-on-signal 15 writes the report
```

`-d` gives per-kernel device time, `-h` host-side API time; the "Device Timing Summary" lands in the
server's stderr. Run the server under the same `prlimit`/`setpriv` wrapper as production if it uses
`QWFN_LOCK_HOST`, and pause/resume as the same user.
