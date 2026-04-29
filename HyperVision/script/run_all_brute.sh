#!/usr/bin/env bash

set -eux

make -j$(nproc)


ARR=(
    "charrdos"
    "cldaprdos"
    "dnsrdos"
    "dnsscan"
    "httpscan"
    "httpsscan"
    "icmpscan"
    "icmpsdos"
    "memcachedrdos"
    "ntprdos"
    "ntpscan"
    "riprdos"
    "rstsdos"
    "sqlscan"
    "ssdprdos"
    "sshscan"
    "synsdos"
    "udpsdos"
)

for item in ${ARR[@]}; do
    ./HyperVision -config ../configuration/bruteforce/${item}.json > ../cache/${item}.log # &
done
