#!/usr/bin/env bash
# Run ALL 43 attacks with KDE-enhanced configs
set -eux

export LD_LIBRARY_PATH="$HOME/local/lib:$HOME/local/lib64:${LD_LIBRARY_PATH:-}"

make -j$(nproc)

mkdir -p ../cache_kde ../temp_kde

# bruteforce_kde (18 attacks)
for item in charrdos cldaprdos dnsrdos dnsscan httpscan httpsscan icmpscan icmpsdos memcachedrdos ntprdos ntpscan riprdos rstsdos sqlscan ssdprdos sshscan synsdos udpsdos; do
    ./HyperVision -config ../configuration/bruteforce_kde/${item}.json > ../cache_kde/${item}.log
done

# lrscan_kde (10 attacks)
for item in dns_lrscan http_lrscan icmp_lrscan netbios_lrscan rdp_lrscan smtp_lrscan snmp_lrscan ssh_lrscan telnet_lrscan vlc_lrscan; do
    ./HyperVision -config ../configuration/lrscan_kde/${item}.json > ../cache_kde/${item}.log
done

# misc_kde (15 attacks)
for item in sshpwdsm sshpwdmd sshpwdla telnetpwdsm telnetpwdmd telnetpwdla crossfiresm crossfiremd crossfirela lrtcpdos02 lrtcpdos05 lrtcpdos10 ackport ipidaddr ipidport; do
    ./HyperVision -config ../configuration/misc_kde/${item}.json > ../cache_kde/${item}.log
done

echo "=== KDE-enhanced: all 43 attacks complete ==="
