#!/usr/bin/env python3
"""
Generate Ground Truth CSV files for HyperVision in PIDSMaker format.

For each of the 43 HyperVision attack files, reads the TSV to find all IPs
involved in attack traffic (label=1) and writes a ground truth CSV.

Ground truth format (3-column, no header):
    node_uuid,label,placeholder
    <IP>,attacker,0
    <IP>,victim,0

Source IPs in attack rows → "attacker"; destination IPs → "victim".
If an IP appears as both src and dst in attack rows, "attacker" takes precedence.

The output filenames must match the entries in config.py's
``ground_truth_relative_path`` and ``attack_to_time_window[i][0]``.

Usage:
    python3 generate_ground_truth_hypervision.py \\
        --input /path/to/hypervision_dataset/

    # Output goes to Ground_Truth/HyperVision/ by default.
"""

import argparse
import csv
import os
import sys

# ── All 43 attack files (same order as create_database_hypervision.py) ─────
# Tuple: (relative_tsv_path, gt_csv_filename)
# gt_csv_filename matches ground_truth_relative_path in config.py.

ATTACK_FILES = [
    # encrypted_flooding_traffic/link_flooding
    ("encrypted_flooding_traffic/link_flooding/crossfiresm.tsv",  "HyperVision/node_crossfiresm.csv"),
    ("encrypted_flooding_traffic/link_flooding/crossfirela.tsv",  "HyperVision/node_crossfirela.csv"),
    ("encrypted_flooding_traffic/link_flooding/crossfiremd.tsv",  "HyperVision/node_crossfiremd.csv"),
    ("encrypted_flooding_traffic/link_flooding/lrtcpdos02.tsv",   "HyperVision/node_lrtcpdos02.csv"),
    ("encrypted_flooding_traffic/link_flooding/lrtcpdos05.tsv",   "HyperVision/node_lrtcpdos05.csv"),
    ("encrypted_flooding_traffic/link_flooding/lrtcpdos10.tsv",   "HyperVision/node_lrtcpdos10.csv"),
    # encrypted_flooding_traffic/password_cracking
    ("encrypted_flooding_traffic/password_cracking/sshpwdla.tsv",     "HyperVision/node_sshpwdla.csv"),
    ("encrypted_flooding_traffic/password_cracking/sshpwdmd.tsv",     "HyperVision/node_sshpwdmd.csv"),
    ("encrypted_flooding_traffic/password_cracking/sshpwdsm.tsv",     "HyperVision/node_sshpwdsm.csv"),
    ("encrypted_flooding_traffic/password_cracking/telnetpwdla.tsv",  "HyperVision/node_telnetpwdla.csv"),
    ("encrypted_flooding_traffic/password_cracking/telnetpwdmd.tsv",  "HyperVision/node_telnetpwdmd.csv"),
    ("encrypted_flooding_traffic/password_cracking/telnetpwdsm.tsv",  "HyperVision/node_telnetpwdsm.csv"),
    # encrypted_flooding_traffic/ssh_inject
    ("encrypted_flooding_traffic/ssh_inject/ackport.tsv",   "HyperVision/node_ackport.csv"),
    ("encrypted_flooding_traffic/ssh_inject/ipidaddr.tsv",  "HyperVision/node_ipidaddr.csv"),
    ("encrypted_flooding_traffic/ssh_inject/ipidport.tsv",  "HyperVision/node_ipidport.csv"),
    # traditional_brute_force_attack/amplification_attack
    ("traditional_brute_force_attack/amplification_attack/charrdos.tsv",      "HyperVision/node_charrdos.csv"),
    ("traditional_brute_force_attack/amplification_attack/cldaprdos.tsv",     "HyperVision/node_cldaprdos.csv"),
    ("traditional_brute_force_attack/amplification_attack/dnsrdos.tsv",       "HyperVision/node_dnsrdos.csv"),
    ("traditional_brute_force_attack/amplification_attack/memcachedrdos.tsv", "HyperVision/node_memcachedrdos.csv"),
    ("traditional_brute_force_attack/amplification_attack/ntprdos.tsv",       "HyperVision/node_ntprdos.csv"),
    ("traditional_brute_force_attack/amplification_attack/riprdos.tsv",       "HyperVision/node_riprdos.csv"),
    ("traditional_brute_force_attack/amplification_attack/ssdprdos.tsv",      "HyperVision/node_ssdprdos.csv"),
    # traditional_brute_force_attack/brute_scanning
    ("traditional_brute_force_attack/brute_scanning/dnsscan.tsv",   "HyperVision/node_dnsscan.csv"),
    ("traditional_brute_force_attack/brute_scanning/httpscan.tsv",  "HyperVision/node_httpscan.csv"),
    ("traditional_brute_force_attack/brute_scanning/httpsscan.tsv", "HyperVision/node_httpsscan.csv"),
    ("traditional_brute_force_attack/brute_scanning/icmpscan.tsv",  "HyperVision/node_icmpscan.csv"),
    ("traditional_brute_force_attack/brute_scanning/ntpscan.tsv",   "HyperVision/node_ntpscan.csv"),
    ("traditional_brute_force_attack/brute_scanning/sqlscan.tsv",   "HyperVision/node_sqlscan.csv"),
    ("traditional_brute_force_attack/brute_scanning/sshscan.tsv",   "HyperVision/node_sshscan.csv"),
    # traditional_brute_force_attack/probing_vulnerable_application
    ("traditional_brute_force_attack/probing_vulnerable_application/dns_lrscan.tsv",     "HyperVision/node_dns_lrscan.csv"),
    ("traditional_brute_force_attack/probing_vulnerable_application/http_lrscan.tsv",    "HyperVision/node_http_lrscan.csv"),
    ("traditional_brute_force_attack/probing_vulnerable_application/icmp_lrscan.tsv",    "HyperVision/node_icmp_lrscan.csv"),
    ("traditional_brute_force_attack/probing_vulnerable_application/netbios_lrscan.tsv", "HyperVision/node_netbios_lrscan.csv"),
    ("traditional_brute_force_attack/probing_vulnerable_application/rdp_lrscan.tsv",     "HyperVision/node_rdp_lrscan.csv"),
    ("traditional_brute_force_attack/probing_vulnerable_application/smtp_lrscan.tsv",    "HyperVision/node_smtp_lrscan.csv"),
    ("traditional_brute_force_attack/probing_vulnerable_application/snmp_lrscan.tsv",    "HyperVision/node_snmp_lrscan.csv"),
    ("traditional_brute_force_attack/probing_vulnerable_application/ssh_lrscan.tsv",     "HyperVision/node_ssh_lrscan.csv"),
    ("traditional_brute_force_attack/probing_vulnerable_application/telnet_lrscan.tsv",  "HyperVision/node_telnet_lrscan.csv"),
    ("traditional_brute_force_attack/probing_vulnerable_application/vlc_lrscan.tsv",     "HyperVision/node_vlc_lrscan.csv"),
    # traditional_brute_force_attack/source_spoof
    ("traditional_brute_force_attack/source_spoof/icmpsdos.tsv",  "HyperVision/node_icmpsdos.csv"),
    ("traditional_brute_force_attack/source_spoof/rstsdos.tsv",   "HyperVision/node_rstsdos.csv"),
    ("traditional_brute_force_attack/source_spoof/synsdos.tsv",   "HyperVision/node_synsdos.csv"),
    ("traditional_brute_force_attack/source_spoof/udpsdos.tsv",   "HyperVision/node_udpsdos.csv"),
]


def extract_attack_ips(tsv_path):
    """Read a HyperVision TSV and return (attacker_ips, victim_ips) sets.

    Source IPs in attack rows (label=1) → attacker_ips.
    Destination IPs in attack rows → victim_ips.
    """
    attackers = set()
    victims = set()
    with open(tsv_path, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            label = parts[7].strip()
            if label != "1":
                continue
            src_ip = parts[0].strip()
            dst_ip = parts[2].strip()
            attackers.add(src_ip)
            victims.add(dst_ip)
    return attackers, victims


def main():
    parser = argparse.ArgumentParser(
        description="Generate HyperVision ground truth for PIDSMaker"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to hypervision_dataset/ root directory",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for ground truth CSVs "
             "(default: <PIDSMaker>/Ground_Truth/)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_base = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_base = os.path.join(script_dir, "..", "..", "Ground_Truth")

    print(f"[*] Input directory: {os.path.abspath(args.input)}")
    print(f"[*] Output directory: {os.path.abspath(output_base)}")

    total_files = 0
    total_nodes = 0

    for tsv_rel, gt_csv in ATTACK_FILES:
        tsv_path = os.path.join(args.input, tsv_rel)
        if not os.path.exists(tsv_path):
            print(f"[!] Missing: {tsv_path}")
            continue

        attackers, victims = extract_attack_ips(tsv_path)

        out_path = os.path.join(output_base, gt_csv)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        written = 0
        with open(out_path, "w") as f:
            # Attackers first (src IPs take precedence)
            for ip in sorted(attackers):
                f.write(f"{ip},attacker,0\n")
                written += 1
            # Victims that are not also attackers
            for ip in sorted(victims - attackers):
                f.write(f"{ip},victim,0\n")
                written += 1

        attack_name = os.path.splitext(os.path.basename(tsv_rel))[0]
        print(f"    [+] {gt_csv} — {len(attackers)} attackers, "
              f"{len(victims - attackers)} victims ({written} total)")
        total_files += 1
        total_nodes += written

    print(f"[+] {total_files} ground truth files, "
          f"{total_nodes} total malicious nodes")
    print(f"[+] Saved to: {os.path.abspath(output_base)}/HyperVision/")


if __name__ == "__main__":
    main()
