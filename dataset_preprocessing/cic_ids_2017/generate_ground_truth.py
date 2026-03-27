#!/usr/bin/env python3
"""
Generate Ground Truth CSV files for CIC-IDS-2017 in PIDSMaker format.

Ground truth = all nodes in the attack chain (attackers AND victims).

Attacker nodes:
  - 172.16.0.1  (firewall/NAT) — always marked malicious; represents external
    attackers after NAT for most attacks.
  - 205.174.165.73  (Kali) — external attacker IP, visible in the dataset for
    some attacks (Infiltration step1, Botnet ARES) and included in all
    externally-sourced attacks per official CIC-IDS-2017 documentation.
  - 192.168.10.8  (Win Vista) — becomes an attacker during Thursday Infiltration
    second step (portscan from Vista to all other internal clients).
  - 205.174.165.69/70/71  (external Win machines) — DDoS LOIT attackers,
    NAT'd through 172.16.0.1.

Victim nodes are marked malicious as part of the full attack chain,
based on official CIC-IDS-2017 dataset documentation.

Output: CSV files with columns matching PIDSMaker ground truth format:
    node_uuid, label, placeholder  (3-column format, node_uuid = IP address)
    label is "attacker" for attacker nodes, "victim" for victim nodes.

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
# Format: (gt_filename, [(ip, label), ...])
#
# Full attack chain: both attacker and victim nodes are included.
# label is "attacker" or "victim".
#
# Attacker IPs:
#   172.16.0.1      = firewall/NAT (always malicious)
#   205.174.165.73  = Kali (external attacker)
#   192.168.10.8    = Vista (becomes attacker in Infiltration step2)
#   205.174.165.69/70/71 = external Win machines (DDoS LOIT)
#
# Victim IPs: per official CIC-IDS-2017 documentation.
# CSV flow verification where data is available; official mapping used
# for attacks whose time windows fall outside the truncated CSV coverage.

ATTACK_GROUND_TRUTH = [
    # ── Tuesday ────────────────────────────────────────────────────────────
    # FTP-Patator (9:20-10:20 EDT): Kali -> firewall -> WebServer
    # CSV-verified: 172.16.0.1 <-> 192.168.10.50 (487 flows)
    ("CIC-IDS-2017/node_ftp_patator_tue.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.50", "victim"),
    ]),

    # SSH-Patator (14:00-15:00 EDT): Kali -> firewall -> WebServer
    # CSV truncated at 13:32 EDT; official victim = WebServer 192.168.10.50
    ("CIC-IDS-2017/node_ssh_patator_tue.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.50", "victim"),
    ]),

    # ── Wednesday ──────────────────────────────────────────────────────────
    # DoS Slowloris (9:47-10:10 EDT): Kali -> firewall -> WebServer
    # CSV-verified: 172.16.0.1 <-> 192.168.10.50, .51 (1,778,079 flows)
    ("CIC-IDS-2017/node_dos_slowloris_wed.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.50", "victim"),
    ]),

    # DoS Slowhttptest (10:14-10:35 EDT): Kali -> firewall -> WebServer
    # CSV partially covers (ends 10:18); verified 172.16.0.1 <-> .50,.51
    # Official victim = WebServer 192.168.10.50
    ("CIC-IDS-2017/node_dos_slowhttptest_wed.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.50", "victim"),
    ]),

    # DoS Hulk (10:43-11:00 EDT): Kali -> firewall -> WebServer
    # CSV truncated at 10:18 EDT; official victim = WebServer 192.168.10.50
    ("CIC-IDS-2017/node_dos_hulk_wed.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.50", "victim"),
    ]),

    # DoS GoldenEye (11:10-11:23 EDT): Kali -> firewall -> WebServer
    # CSV truncated at 10:18 EDT; official victim = WebServer 192.168.10.50
    ("CIC-IDS-2017/node_dos_goldeneye_wed.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.50", "victim"),
    ]),

    # Heartbleed (15:12-15:32 EDT): Kali -> firewall -> Ubuntu12
    # CSV truncated at 10:18 EDT; official victim = Ubuntu12 192.168.10.51
    ("CIC-IDS-2017/node_heartbleed_wed.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.51", "victim"),
    ]),

    # ── Thursday AM ────────────────────────────────────────────────────────
    # Web Brute Force (9:20-10:00 EDT): Kali -> firewall -> WebServer
    # CSV-verified: 172.16.0.1 <-> 192.168.10.50 (7,826 flows)
    ("CIC-IDS-2017/node_web_bruteforce_thu.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.50", "victim"),
    ]),

    # Web XSS (10:15-10:35 EDT): Kali -> firewall -> WebServer
    # App-layer attack; no direct attacker flows visible at flow level.
    # Official victim = WebServer 192.168.10.50
    ("CIC-IDS-2017/node_web_xss_thu.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.50", "victim"),
    ]),

    # Web SQL Injection (10:40-10:42 EDT): Kali -> firewall -> WebServer
    # App-layer attack; no direct attacker flows visible at flow level.
    # Official victim = WebServer 192.168.10.50
    ("CIC-IDS-2017/node_web_sqli_thu.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.50", "victim"),
    ]),

    # ── Thursday PM ────────────────────────────────────────────────────────
    # Infiltration step1: Metasploit (14:19-14:35 EDT)
    # Kali -> Vista via Dropbox exploit (bypasses NAT)
    # CSV-verified: 205.174.165.73 <-> 192.168.10.8 (349 flows)
    ("CIC-IDS-2017/node_infiltration_step1_thu.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.8", "victim"),
    ]),

    # Infiltration Cool Disk – MAC (14:53-15:00 EDT)
    # Kali -> MAC via removable media; no network flows expected.
    # Official: attacker=Kali, victim=MAC 192.168.10.25
    ("CIC-IDS-2017/node_infiltration_cooldisk_thu.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.25", "victim"),
    ]),

    # Infiltration step2: Vista portscan (15:04-15:45 EDT)
    # Compromised Vista scans all other internal clients.
    # CSV partially covers (ends 15:21); official victim = all other clients.
    ("CIC-IDS-2017/node_infiltration_step2_thu.csv", [
        ("192.168.10.8", "attacker"),
        ("192.168.10.3", "victim"),     # DNS+DC Server
        ("192.168.10.5", "victim"),     # Win 8.1
        ("192.168.10.9", "victim"),     # Win 7
        ("192.168.10.12", "victim"),    # Ubuntu 16.4 64B
        ("192.168.10.14", "victim"),    # Win 10 pro 32B
        ("192.168.10.15", "victim"),    # Win 10 64B
        ("192.168.10.16", "victim"),    # Ubuntu 16.4 32B
        ("192.168.10.17", "victim"),    # Ubuntu 14.4 64B
        ("192.168.10.19", "victim"),    # Ubuntu 14.4 32B
        ("192.168.10.25", "victim"),    # MAC
        ("192.168.10.50", "victim"),    # Web server 16
        ("192.168.10.51", "victim"),    # Ubuntu server 12
    ]),

    # ── Friday AM ──────────────────────────────────────────────────────────
    # Botnet ARES (10:02-11:02 EDT): Kali -> 5 Windows/Vista victims
    # CSV-verified: 205.174.165.73 <-> all 5 victims (1,479 flows)
    ("CIC-IDS-2017/node_botnet_fri.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.5", "victim"),     # Win 8.1
        ("192.168.10.8", "victim"),     # Vista
        ("192.168.10.9", "victim"),     # Win 7
        ("192.168.10.14", "victim"),    # Win 10 pro 32B
        ("192.168.10.15", "victim"),    # Win 10 64B
    ]),

    # ── Friday PM ──────────────────────────────────────────────────────────
    # Port Scan (13:55-15:29 EDT): Kali -> firewall -> WebServer
    # CSV-verified: 172.16.0.1 <-> 192.168.10.50 (425,489 flows)
    ("CIC-IDS-2017/node_portscan_fri.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.73", "attacker"),
        ("192.168.10.50", "victim"),
    ]),

    # DDoS LOIT (15:56-16:16 EDT): 3 external Win -> firewall -> WebServer
    # CSV truncated at 15:01 EDT; official victim = WebServer 192.168.10.50.
    # External attackers 205.174.165.69/70/71 NAT'd through 172.16.0.1.
    ("CIC-IDS-2017/node_ddos_loit_fri.csv", [
        ("172.16.0.1", "attacker"),
        ("205.174.165.69", "attacker"),
        ("205.174.165.70", "attacker"),
        ("205.174.165.71", "attacker"),
        ("192.168.10.50", "victim"),
    ]),
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
    key_ips = ["172.16.0.1", "205.174.165.73", "192.168.10.8",
               "205.174.165.69", "205.174.165.70", "205.174.165.71"]
    for ip in key_ips:
        if ip in ip_to_index:
            print(f"    {ip} -> index_id {ip_to_index[ip]}")
        else:
            print(f"    [!] WARNING: {ip} not found in dataset (will skip in GT files)")

    # Generate ground truth CSVs
    print("[*] Generating ground truth CSV files...")
    total_nodes = 0
    for gt_filename, ip_label_pairs in ATTACK_GROUND_TRUTH:
        out_path = os.path.join(output_base, gt_filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        written = 0
        with open(out_path, "w") as f:
            for ip, label in ip_label_pairs:
                if ip in ip_to_index:
                    # Write 3-column format: node_uuid (IP address), label, placeholder
                    # node_uuid must match what create_database_cic_ids_2017.py stores
                    # in netflow_node_table.node_uuid (which is the IP address itself).
                    f.write(f"{ip},{label},0\n")
                    written += 1
                else:
                    print(f"    [!] WARNING: {ip} not in dataset, skipped for {gt_filename}")

        total_nodes += written
        print(f"    [+] {gt_filename} ({written} nodes)")

    print(f"[+] {len(ATTACK_GROUND_TRUTH)} ground truth files, {total_nodes} total malicious nodes")
    print(f"[+] Saved to: {os.path.abspath(output_base)}/CIC-IDS-2017/")


if __name__ == "__main__":
    main()
