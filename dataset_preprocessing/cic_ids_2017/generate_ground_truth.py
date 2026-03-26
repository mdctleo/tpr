#!/usr/bin/env python3
"""
Generate Ground Truth CSV files for CIC-IDS-2017 in PIDSMaker format.

Ground truth = malicious NODE IDs (attacker-side only, not victims).

Attacker-side nodes:
  - 172.16.0.1 (firewall/NAT) — represents external Kali attacker (205.174.165.73)
    for all externally-sourced attacks.
  - 192.168.10.8 (Win Vista) — becomes an attacker during Thursday Infiltration
    second step (portscan from Vista to all other clients).

Victim nodes are NOT marked malicious unless they pass on the attack.

Output: CSV files with columns matching PIDSMaker ground truth format:
    index_id  (one malicious node ID per line)

Usage:
    python3 generate_ground_truth.py \
        --input /path/to/CIC-IDS-2017/

    # Output goes to Ground_Truth/CIC-IDS-2017/ by default.
    # Both orthrus_cicids and kairos_cicids use ground_truth_version: none,
    # which resolves _ground_truth_dir directly to Ground_Truth/ (no subdirectory).
"""

import argparse
import csv
import hashlib
import os
import sys


def sha256_hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_ip_to_index(input_dir):
    """Build IP-to-index mapping consistent with create_database_cic_ids_2017.py."""
    ip_set = set()
    csv_files = [
        "Monday-WorkingHours.csv",
        "Tuesday-WorkingHours.csv",
        "Wednesday-WorkingHours.csv",
        "Thursday-WorkingHours.csv",
        "Friday-WorkingHours.csv",
    ]
    for csv_file in csv_files:
        path = os.path.join(input_dir, csv_file)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) < 6:
                    continue
                ip_set.add(row[0])
                ip_set.add(row[2])

    ip_to_index = {}
    for idx, ip in enumerate(sorted(ip_set)):
        ip_to_index[ip] = idx

    return ip_to_index


# ── Attack definitions ────────────────────────────────────────────────────────
# Format: (gt_filename, list_of_attacker_IPs)
#
# Only attacker-side nodes are included.
# 172.16.0.1 = external attacker (Kali) after NAT
# 192.168.10.8 = Vista (becomes attacker in Infiltration second step)

ATTACK_GROUND_TRUTH = [
    # Tuesday: FTP-Patator (Kali -> via firewall -> WebServer)
    ("CIC-IDS-2017/node_ftp_patator_tue.csv", ["172.16.0.1"]),

    # Tuesday: SSH-Patator (Kali -> via firewall -> WebServer)
    ("CIC-IDS-2017/node_ssh_patator_tue.csv", ["172.16.0.1"]),

    # Wednesday: DoS Slowloris (Kali -> via firewall -> WebServer)
    ("CIC-IDS-2017/node_dos_slowloris_wed.csv", ["172.16.0.1"]),

    # Wednesday: DoS Slowhttptest (Kali -> via firewall -> WebServer)
    ("CIC-IDS-2017/node_dos_slowhttptest_wed.csv", ["172.16.0.1"]),

    # Wednesday: DoS Hulk (Kali -> via firewall -> WebServer)
    ("CIC-IDS-2017/node_dos_hulk_wed.csv", ["172.16.0.1"]),

    # Wednesday: DoS GoldenEye (Kali -> via firewall -> WebServer)
    ("CIC-IDS-2017/node_dos_goldeneye_wed.csv", ["172.16.0.1"]),

    # Wednesday: Heartbleed (Kali -> via firewall -> Ubuntu12)
    ("CIC-IDS-2017/node_heartbleed_wed.csv", ["172.16.0.1"]),

    # Thursday AM: Web Attack Brute Force (Kali -> via firewall -> WebServer)
    ("CIC-IDS-2017/node_web_bruteforce_thu.csv", ["172.16.0.1"]),

    # Thursday AM: Web Attack XSS (Kali -> via firewall -> WebServer)
    ("CIC-IDS-2017/node_web_xss_thu.csv", ["172.16.0.1"]),

    # Thursday AM: Web Attack SQL Injection (Kali -> via firewall -> WebServer)
    ("CIC-IDS-2017/node_web_sqli_thu.csv", ["172.16.0.1"]),

    # Thursday PM: Infiltration first step (Kali -> Vista via Dropbox/Metasploit)
    # Attack comes through firewall to Vista
    ("CIC-IDS-2017/node_infiltration_step1_thu.csv", ["172.16.0.1"]),

    # Thursday PM: Infiltration second step (Vista does portscan on all clients)
    # Vista is now the ATTACKER
    ("CIC-IDS-2017/node_infiltration_step2_thu.csv", ["192.168.10.8"]),

    # Friday AM: Botnet ARES (Kali -> via firewall -> multiple Windows victims)
    ("CIC-IDS-2017/node_botnet_fri.csv", ["172.16.0.1"]),

    # Friday PM: Port Scan (Kali -> via firewall -> WebServer)
    ("CIC-IDS-2017/node_portscan_fri.csv", ["172.16.0.1"]),

    # Friday PM: DDoS LOIT (3 external Win machines -> via firewall -> WebServer)
    # All 3 attackers go through NAT, appearing as 172.16.0.1
    ("CIC-IDS-2017/node_ddos_loit_fri.csv", ["172.16.0.1"]),
]


def main():
    parser = argparse.ArgumentParser(description="Generate CIC-IDS-2017 ground truth for PIDSMaker")
    parser.add_argument("--input", required=True, help="Path to CIC-IDS-2017 CSV directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for ground truth CSVs (default: Ground_Truth/)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output:
        output_base = args.output
    else:
        # Default: Ground_Truth/ — both orthrus_cicids and kairos_cicids use
        # ground_truth_version: none, which resolves directly to Ground_Truth/.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_base = os.path.join(script_dir, "..", "..", "Ground_Truth")

    # Build IP-to-index mapping
    print("[*] Building IP-to-index mapping from dataset...")
    ip_to_index = get_ip_to_index(args.input)
    print(f"    Total IPs: {len(ip_to_index)}")

    # Check that key attacker IPs exist in dataset
    for ip in ["172.16.0.1", "192.168.10.8"]:
        if ip in ip_to_index:
            print(f"    {ip} -> index_id {ip_to_index[ip]}")
        else:
            print(f"    [!] WARNING: {ip} not found in dataset!")

    # Generate ground truth CSVs
    print("[*] Generating ground truth CSV files...")
    for gt_filename, attacker_ips in ATTACK_GROUND_TRUTH:
        out_path = os.path.join(output_base, gt_filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w") as f:
            for ip in attacker_ips:
                if ip in ip_to_index:
                    f.write(f"{ip_to_index[ip]}\n")
                else:
                    print(f"    [!] WARNING: {ip} not in dataset for {gt_filename}")

        print(f"    [+] {gt_filename}")

    print(f"[+] Ground truth files saved to: {os.path.abspath(output_base)}/CIC-IDS-2017/")


if __name__ == "__main__":
    main()
