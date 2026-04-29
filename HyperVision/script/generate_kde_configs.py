#!/usr/bin/env python3
"""Generate KDE-enhanced JSON configs for all 43 attacks.
Creates configuration/{category}_kde/{attack}.json with kde_features_path added.
Also saves results to temp_kde/ instead of temp/.
"""
import json, os

ATTACKS = {
    "bruteforce": [
        "charrdos", "cldaprdos", "dnsrdos", "dnsscan", "httpscan", "httpsscan",
        "icmpscan", "icmpsdos", "memcachedrdos", "ntprdos", "ntpscan", "riprdos",
        "rstsdos", "sqlscan", "ssdprdos", "sshscan", "synsdos", "udpsdos",
    ],
    "lrscan": [
        "dns_lrscan", "http_lrscan", "icmp_lrscan", "netbios_lrscan", "rdp_lrscan",
        "smtp_lrscan", "snmp_lrscan", "ssh_lrscan", "telnet_lrscan", "vlc_lrscan",
    ],
    "misc": [
        "sshpwdsm", "sshpwdmd", "sshpwdla", "telnetpwdsm", "telnetpwdmd", "telnetpwdla",
        "crossfiresm", "crossfiremd", "crossfirela", "lrtcpdos02", "lrtcpdos05", "lrtcpdos10",
        "ackport", "ipidaddr", "ipidport",
    ],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "..", "configuration")

for category, attacks in ATTACKS.items():
    src_dir = os.path.join(CONFIG_DIR, category)
    dst_dir = os.path.join(CONFIG_DIR, f"{category}_kde")
    os.makedirs(dst_dir, exist_ok=True)

    for attack in attacks:
        src_path = os.path.join(src_dir, f"{attack}.json")
        dst_path = os.path.join(dst_dir, f"{attack}.json")

        with open(src_path, 'r') as f:
            cfg = json.load(f)

        # Add KDE features path to graph_analyze section
        cfg["graph_analyze"]["kde_features_path"] = f"../kde_features/{attack}_kde.csv"

        # Save results to a separate directory
        cfg["result_save"]["save_result_path"] = f"../temp_kde/{attack}.txt"

        with open(dst_path, 'w') as f:
            json.dump(cfg, f, indent=4)

        print(f"  {dst_path}")

print(f"\nGenerated {sum(len(v) for v in ATTACKS.values())} KDE-enhanced configs.")
