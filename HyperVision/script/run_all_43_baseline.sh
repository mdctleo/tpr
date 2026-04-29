#!/usr/bin/env bash
# Run ALL 43 attacks (baseline, no KDE) — bruteforce + lrscan + misc (encrypted_flooding + traditional_brute_force)
set -eux

export LD_LIBRARY_PATH="$HOME/local/lib:$HOME/local/lib64:${LD_LIBRARY_PATH:-}"

make -j$(nproc)

mkdir -p ../cache ../temp

# bruteforce (18 attacks: amplification + brute_scanning + source_spoof)
for item in charrdos cldaprdos dnsrdos dnsscan httpscan httpsscan icmpscan icmpsdos memcachedrdos ntprdos ntpscan riprdos rstsdos sqlscan ssdprdos sshscan synsdos udpsdos; do
    ./HyperVision -config ../configuration/bruteforce/${item}.json > ../cache/${item}.log
done

# lrscan (10 attacks: probing_vulnerable_application)
for item in dns_lrscan http_lrscan icmp_lrscan netbios_lrscan rdp_lrscan smtp_lrscan snmp_lrscan ssh_lrscan telnet_lrscan vlc_lrscan; do
    ./HyperVision -config ../configuration/lrscan/${item}.json > ../cache/${item}.log
done

# misc (15 attacks: link_flooding + password_cracking + ssh_inject)
for item in sshpwdsm sshpwdmd sshpwdla telnetpwdsm telnetpwdmd telnetpwdla crossfiresm crossfiremd crossfirela lrtcpdos02 lrtcpdos05 lrtcpdos10 ackport ipidaddr ipidport; do
    ./HyperVision -config ../configuration/misc/${item}.json > ../cache/${item}.log
done

echo "=== Baseline: all 43 attacks complete ==="
