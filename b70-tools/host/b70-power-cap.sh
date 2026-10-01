#!/bin/bash
# Apply a power cap to the Intel Arc Pro B70 (run as root; b70-power-cap.timer re-applies it every 10 minutes,
# because the cap resets to the stock 230 W whenever the card is re-initialized, e.g. handed back from a VM).
# The cap: B70_CAP_W (default 160). Measured on the reference machine at 80K: 140 W costs ~6% prefill and decode
# and cuts fan speed ~7%; 130 W costs 13-15%; 230 W buys ~2% over 160 W.
CAP_W=${B70_CAP_W:-160}
B70_ID="8086:e223"
bdf=$(lspci -d $B70_ID 2>/dev/null | awk '{print $1}' | sed 's/^0000://' | head -1)
[ -n "$bdf" ] || exit 0   # not on the host (e.g. passed through to a VM)
for cap in /sys/bus/pci/devices/0000:$bdf/hwmon/hwmon*/power1_cap; do
    [ -f "$cap" ] && echo $((CAP_W * 1000000)) > "$cap" 2>/dev/null
done
exit 0
