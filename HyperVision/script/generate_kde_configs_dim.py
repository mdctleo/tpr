#!/usr/bin/env python3
"""Generate KDE-enhanced JSON configs for all 43 attacks at a given KDE dimension.

Usage:
    python generate_kde_configs_dim.py --kde-dim 10

Creates:
    configuration/{category}_kde_{D}/{attack}.json
    with kde_features_path -> ../kde_features_{D}/{attack}_kde.csv
    and  save_result_path  -> ../temp_kde_{D}/{attack}.txt
"""
import argparse
import json
import os

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


def main():
    parser = argparse.ArgumentParser(description="Generate KDE configs for a given dimension")
    parser.add_argument("--kde-dim", type=int, required=True, help="KDE dimension (e.g. 5,10,15,25,30)")
    args = parser.parse_args()
    D = args.kde_dim

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    config_dir = os.path.join(base_dir, "configuration")
    count = 0

    for category, attacks in ATTACKS.items():
        src_dir = os.path.join(config_dir, category)
        dst_dir = os.path.join(config_dir, f"{category}_kde_{D}")
        os.makedirs(dst_dir, exist_ok=True)

        for attack in attacks:
            src_path = os.path.join(src_dir, f"{attack}.json")
            dst_path = os.path.join(dst_dir, f"{attack}.json")

            with open(src_path, "r") as f:
                cfg = json.load(f)

            cfg["graph_analyze"]["kde_features_path"] = f"../kde_features_{D}/{attack}_kde.csv"
            cfg["result_save"]["save_result_path"] = f"../temp_kde_{D}/{attack}.txt"

            with open(dst_path, "w") as f:
                json.dump(cfg, f, indent=4)

            count += 1

    print(f"Generated {count} configs for kde_dim={D} in configuration/{{cat}}_kde_{D}/")


if __name__ == "__main__":
    main()
