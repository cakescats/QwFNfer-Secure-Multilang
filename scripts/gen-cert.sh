#!/usr/bin/env bash
# Self-signed TLS certificate for the console on the local network.
#
#   scripts/gen-cert.sh [dir]        (default: ~/.cache/qwfn-console/tls)
#
# The certificate names localhost, this machine's hostname and every IPv4 address it has
# right now, so the browser accepts it for any of them once you trust it (the first visit
# shows a warning: check the fingerprint printed below, then accept).
# Then:  scripts/console.sh --host 0.0.0.0 --tls-cert DIR/cert.pem --tls-key DIR/key.pem
set -eu
dir=${1:-${QWFN_CONSOLE_DIR:-$HOME/.cache/qwfn-console}/tls}
mkdir -p "$dir"; chmod 700 "$dir"
san="DNS:localhost,DNS:$(hostname),IP:127.0.0.1"
for ip in $(hostname -I 2>/dev/null); do case $ip in *:*) ;; *) san="$san,IP:$ip" ;; esac; done
umask 077
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 825 \
    -subj "/CN=qwfn console ($(hostname))" -addext "subjectAltName=$san" \
    -keyout "$dir/key.pem" -out "$dir/cert.pem" 2>/dev/null
echo "certificate: $dir/cert.pem"
echo "key:         $dir/key.pem"
echo "names:       $san"
openssl x509 -in "$dir/cert.pem" -noout -fingerprint -sha256
