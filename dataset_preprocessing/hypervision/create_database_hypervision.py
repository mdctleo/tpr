#!/usr/bin/env python3
"""
HyperVision Dataset Ingestion Script for PIDSMaker

Reads the 43 HyperVision TSV files and inserts them into PostgreSQL using
PIDSMaker's event_table / netflow_node_table schema.

Day mapping (44 days, year_month="2024-01"):
  Day  1       = training: benign-only rows from ONE representative file
  Days  2 –  8 = validation: 7 attacks (one per subclass) — full rows
  Days  9 – 44 = testing: remaining 36 attacks — full rows

Each day is committed immediately after COPY so progress is never lost.
Use --resume-from-day to skip already-ingested days after a crash.

Phase 1+2 results (ip_to_index mapping) are cached to
  <input>/prepared/ip_to_index.json
so they are never recomputed.

Usage:
    # Fresh run
    python3 create_database_hypervision.py \\
        --input /path/to/hypervision_dataset/ \\
        --db-name hypervision --create-db

    # Resume after crash at day 15
    python3 create_database_hypervision.py \\
        --input /path/to/hypervision_dataset/ \\
        --db-name hypervision --resume-from-day 15
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("[!] Missing psycopg2. Install with: pip install psycopg2-binary")
    sys.exit(1)

try:
    import pytz
except ImportError:
    print("[!] Missing pytz. Install with: pip install pytz")
    sys.exit(1)


# ── Schema ─────────────────────────────────────────────────────────────────

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS event_table (
    src_node      VARCHAR(64),
    src_index_id  VARCHAR(16),
    operation     VARCHAR(32),
    dst_node      VARCHAR(64),
    dst_index_id  VARCHAR(16),
    event_uuid    VARCHAR(64),
    timestamp_rec BIGINT,
    _id           SERIAL PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS netflow_node_table (
    node_uuid  VARCHAR(64),
    hash_id    VARCHAR(64),
    src_addr   VARCHAR(64),
    src_port   VARCHAR(16),
    dst_addr   VARCHAR(64),
    dst_port   VARCHAR(16),
    index_id   BIGINT
);
CREATE TABLE IF NOT EXISTS subject_node_table (
    node_uuid  VARCHAR(64),
    hash_id    VARCHAR(64),
    path       VARCHAR(256),
    cmd        VARCHAR(256),
    index_id   BIGINT
);
CREATE TABLE IF NOT EXISTS file_node_table (
    node_uuid  VARCHAR(64),
    hash_id    VARCHAR(64),
    path       VARCHAR(256),
    index_id   BIGINT
);
GRANT ALL ON ALL TABLES IN SCHEMA public TO postgres;
"""

# ── Day mapping ────────────────────────────────────────────────────────────
# Day 1 = training (benign only from first file)
# Days 2-8 = validation (one attack per subclass)
# Days 9-44 = testing (remaining 36 attacks)

# The file used for Day 1 benign training data (first val file)
DAY1_BENIGN_SOURCE = "encrypted_flooding_traffic/link_flooding/crossfiresm.tsv"

ATTACK_DAY_MAP = [
    # ── Validation (days 2-8) ──────────────────────────────────────────────
    (2,  "encrypted_flooding_traffic/link_flooding/crossfiresm.tsv"),
    (3,  "encrypted_flooding_traffic/password_cracking/sshpwdla.tsv"),
    (4,  "encrypted_flooding_traffic/ssh_inject/ackport.tsv"),
    (5,  "traditional_brute_force_attack/amplification_attack/charrdos.tsv"),
    (6,  "traditional_brute_force_attack/brute_scanning/sqlscan.tsv"),
    (7,  "traditional_brute_force_attack/probing_vulnerable_application/ssh_lrscan.tsv"),
    (8,  "traditional_brute_force_attack/source_spoof/icmpsdos.tsv"),
    # ── Testing (days 9-44) ────────────────────────────────────────────────
    (9,  "encrypted_flooding_traffic/link_flooding/crossfirela.tsv"),
    (10, "encrypted_flooding_traffic/link_flooding/crossfiremd.tsv"),
    (11, "encrypted_flooding_traffic/link_flooding/lrtcpdos02.tsv"),
    (12, "encrypted_flooding_traffic/link_flooding/lrtcpdos05.tsv"),
    (13, "encrypted_flooding_traffic/link_flooding/lrtcpdos10.tsv"),
    (14, "encrypted_flooding_traffic/password_cracking/sshpwdmd.tsv"),
    (15, "encrypted_flooding_traffic/password_cracking/sshpwdsm.tsv"),
    (16, "encrypted_flooding_traffic/password_cracking/telnetpwdla.tsv"),
    (17, "encrypted_flooding_traffic/password_cracking/telnetpwdmd.tsv"),
    (18, "encrypted_flooding_traffic/password_cracking/telnetpwdsm.tsv"),
    (19, "encrypted_flooding_traffic/ssh_inject/ipidaddr.tsv"),
    (20, "encrypted_flooding_traffic/ssh_inject/ipidport.tsv"),
    (21, "traditional_brute_force_attack/amplification_attack/cldaprdos.tsv"),
    (22, "traditional_brute_force_attack/amplification_attack/dnsrdos.tsv"),
    (23, "traditional_brute_force_attack/amplification_attack/memcachedrdos.tsv"),
    (24, "traditional_brute_force_attack/amplification_attack/ntprdos.tsv"),
    (25, "traditional_brute_force_attack/amplification_attack/riprdos.tsv"),
    (26, "traditional_brute_force_attack/amplification_attack/ssdprdos.tsv"),
    (27, "traditional_brute_force_attack/brute_scanning/dnsscan.tsv"),
    (28, "traditional_brute_force_attack/brute_scanning/httpscan.tsv"),
    (29, "traditional_brute_force_attack/brute_scanning/httpsscan.tsv"),
    (30, "traditional_brute_force_attack/brute_scanning/icmpscan.tsv"),
    (31, "traditional_brute_force_attack/brute_scanning/ntpscan.tsv"),
    (32, "traditional_brute_force_attack/brute_scanning/sshscan.tsv"),
    (33, "traditional_brute_force_attack/probing_vulnerable_application/dns_lrscan.tsv"),
    (34, "traditional_brute_force_attack/probing_vulnerable_application/http_lrscan.tsv"),
    (35, "traditional_brute_force_attack/probing_vulnerable_application/icmp_lrscan.tsv"),
    (36, "traditional_brute_force_attack/probing_vulnerable_application/netbios_lrscan.tsv"),
    (37, "traditional_brute_force_attack/probing_vulnerable_application/rdp_lrscan.tsv"),
    (38, "traditional_brute_force_attack/probing_vulnerable_application/smtp_lrscan.tsv"),
    (39, "traditional_brute_force_attack/probing_vulnerable_application/snmp_lrscan.tsv"),
    (40, "traditional_brute_force_attack/probing_vulnerable_application/telnet_lrscan.tsv"),
    (41, "traditional_brute_force_attack/probing_vulnerable_application/vlc_lrscan.tsv"),
    (42, "traditional_brute_force_attack/source_spoof/rstsdos.tsv"),
    (43, "traditional_brute_force_attack/source_spoof/synsdos.tsv"),
    (44, "traditional_brute_force_attack/source_spoof/udpsdos.tsv"),
]

VALID_EDGE_TYPES = {
    "ICMP", "TCP_ACK", "TCP_ACK+FIN", "TCP_ACK+RST", "TCP_RST",
    "TCP_SYN", "TCP_SYN+ACK", "UDP", "UNKNOWN",
}


def sha256_hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _day_base_ns(day):
    """Epoch nanoseconds for start of logical day (timedelta handles day>31)."""
    tz = pytz.timezone("US/Eastern")
    base = datetime(2024, 1, 1)
    dt = base + timedelta(days=day - 1)
    localized = tz.localize(dt)
    return int(localized.timestamp() * 1_000_000_000)


def _connect_with_retry(conn_params, max_retries=5, base_delay=5):
    """Connect to PostgreSQL with exponential back-off retries."""
    for attempt in range(1, max_retries + 1):
        try:
            return psycopg2.connect(**conn_params)
        except psycopg2.OperationalError as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            print(f"\n    [!] Connection failed (attempt {attempt}/{max_retries}): "
                  f"{e}\n    Retrying in {delay}s...", flush=True)
            time.sleep(delay)


# ── Subprocess worker ──────────────────────────────────────────────────────
# Converts a HyperVision TSV into a tab-separated COPY-ready file.
# Accepts an optional 6th arg "benign" to keep only label=0 rows.
# Runs in a fresh process to avoid psycopg2 C-extension memory issues.

_WORKER_SCRIPT = r"""
import hashlib, json, sys

with open(sys.argv[1]) as jf:
    ip_to_index = json.load(jf)

tsv_path    = sys.argv[2]
out_path    = sys.argv[3]
evt_offset  = int(sys.argv[4])
day_base_ns = int(sys.argv[5])
benign_only = len(sys.argv) > 6 and sys.argv[6] == "benign"

VALID = {"ICMP", "TCP_ACK", "TCP_ACK+FIN", "TCP_ACK+RST", "TCP_RST",
         "TCP_SYN", "TCP_SYN+ACK", "UDP", "UNKNOWN"}

n = 0
with open(out_path, "wb") as out, open(tsv_path, "rb") as f:
    next(f)  # skip header
    for raw in f:
        p = raw.replace(b"\x00", b"").split(b"\t")
        if len(p) < 8:
            continue
        if benign_only and p[7].strip() != b"0":
            continue
        src_ip = p[0].decode("utf-8", "replace").strip()
        dst_ip = p[2].decode("utf-8", "replace").strip()
        op     = p[4].decode("utf-8", "replace").strip()
        ts_us  = p[5].decode("utf-8", "replace").strip()
        if op not in VALID:
            continue
        si = ip_to_index.get(src_ip)
        di = ip_to_index.get(dst_ip)
        if si is None or di is None:
            continue
        try:
            ts_ns = day_base_ns + int(ts_us) * 1000
        except Exception:
            continue
        sh = hashlib.sha256(src_ip.encode()).hexdigest()
        dh = hashlib.sha256(dst_ip.encode()).hexdigest()
        line = f"{sh}\t{si}\t{op}\t{dh}\t{di}\tevt_{evt_offset + n}\t{ts_ns}\n"
        out.write(line.encode("utf-8"))
        n += 1

print(n)
"""


# ── Helper: run worker, COPY result, commit ────────────────────────────────

def _ingest_one_day(conn_params, idx_path, tsv_path, day, event_offset,
                    benign_only=False):
    """Convert TSV -> temp file via subprocess, COPY into DB, commit.

    Returns (num_events, elapsed_seconds).
    """
    day_ns = _day_base_ns(day)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tsv", prefix=f"hv_day{day}_")
    os.close(tmp_fd)

    try:
        # ── Subprocess: TSV -> COPY-ready file ─────────────────────────
        cmd = [
            sys.executable, "-c", _WORKER_SCRIPT,
            idx_path, tsv_path, tmp_path,
            str(event_offset), str(day_ns),
        ]
        if benign_only:
            cmd.append("benign")

        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f" ERROR (rc={result.returncode})")
            if result.stderr:
                print(f"      stderr: {result.stderr[:400]}")
            return 0, time.time() - t0

        n = int(result.stdout.strip())
        if n == 0:
            print(f" 0 events (empty)")
            return 0, time.time() - t0

        # ── COPY into DB and commit immediately ───────────────────────
        conn = _connect_with_retry(conn_params)
        cur = conn.cursor()
        try:
            with open(tmp_path, "rb") as f:
                cur.copy_expert(
                    "COPY event_table (src_node, src_index_id, operation, "
                    "dst_node, dst_index_id, event_uuid, timestamp_rec) "
                    "FROM STDIN",
                    f,
                )
            conn.commit()
        finally:
            cur.close()
            conn.close()

        elapsed = time.time() - t0
        return n, elapsed

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Phase 1+2: IP collection and netflow node insertion ───────────────────

def ensure_ip_index(input_dir, all_tsv_paths):
    """Scan all TSVs for unique IPs. Cache result to prepared/ip_to_index.json."""
    cache_dir = os.path.join(input_dir, "prepared")
    cache_path = os.path.join(cache_dir, "ip_to_index.json")

    if os.path.exists(cache_path):
        print(f"[*] Phase 1: Loading cached IP index from {cache_path}")
        with open(cache_path) as f:
            ip_to_index = json.load(f)
        print(f"    {len(ip_to_index)} IPs loaded from cache")
        return ip_to_index, cache_path

    print("[*] Phase 1: Scanning all 43 TSV files for unique IPs...")
    ip_set = set()
    for i, path in enumerate(all_tsv_paths):
        print(f"    [{i+1}/{len(all_tsv_paths)}] {os.path.basename(path)}...",
              end="", flush=True)
        with open(path, "rb") as f:
            next(f)
            for raw in f:
                parts = raw.replace(b"\x00", b"").split(b"\t")
                if len(parts) >= 3:
                    ip_set.add(parts[0].decode("utf-8", "replace").strip())
                    ip_set.add(parts[2].decode("utf-8", "replace").strip())
        print(" done")
    ip_set.discard("")
    print(f"    Found {len(ip_set)} unique IP addresses")

    ip_to_index = {ip: idx for idx, ip in enumerate(sorted(ip_set))}

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(ip_to_index, f)
    print(f"    Cached to {cache_path}")

    return ip_to_index, cache_path


def insert_netflow_nodes(conn_params, ip_to_index):
    """Insert IPs into netflow_node_table (Phase 2) using bulk COPY."""
    conn = _connect_with_retry(conn_params)
    cur = conn.cursor()
    try:
        # Check if already populated
        cur.execute("SELECT COUNT(*) FROM netflow_node_table")
        existing = cur.fetchone()[0]
        if existing == len(ip_to_index):
            print(f"[*] Phase 2: netflow_node_table already has "
                  f"{existing} nodes — skipping")
            return
        elif existing > 0:
            print(f"[*] Phase 2: Truncating stale netflow_node_table "
                  f"({existing} rows)...")
            cur.execute("TRUNCATE TABLE netflow_node_table")
            conn.commit()

        print(f"[*] Phase 2: Inserting {len(ip_to_index)} netflow nodes "
              f"via bulk COPY...")
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".tsv", prefix="hv_netflow_")
        os.close(tmp_fd)
        try:
            with open(tmp_path, "w") as f:
                for ip, idx in sorted(
                    ip_to_index.items(), key=lambda x: x[1]
                ):
                    h = sha256_hash(ip)
                    f.write(f"{ip}\t{h}\t{ip}\t0\t{ip}\t0\t{idx}\n")
            with open(tmp_path, "rb") as f:
                cur.copy_expert(
                    "COPY netflow_node_table "
                    "(node_uuid, hash_id, src_addr, src_port, "
                    "dst_addr, dst_port, index_id) FROM STDIN",
                    f,
                )
            conn.commit()
            print(f"    Inserted {len(ip_to_index)} netflow nodes")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    finally:
        cur.close()
        conn.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest HyperVision dataset into PostgreSQL for PIDSMaker"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to hypervision_dataset/ root directory",
    )
    parser.add_argument("--db-name", default="hypervision")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument(
        "--create-db", action="store_true",
        help="Create database if it doesn't exist",
    )
    parser.add_argument(
        "--resume-from-day", type=int, default=0,
        help="Skip days before this number (use after a crash). "
             "E.g. --resume-from-day 15 skips days 1-14.",
    )
    args = parser.parse_args()

    conn_params = dict(
        dbname=args.db_name, user=args.user, password=args.password,
        host=args.host, port=args.port,
    )

    # ── Create DB if needed ────────────────────────────────────────────
    if args.create_db:
        admin_conn = psycopg2.connect(
            dbname="postgres", user=args.user, password=args.password,
            host=args.host, port=args.port,
        )
        admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = admin_conn.cursor()
        cur.execute(
            f"SELECT 1 FROM pg_database WHERE datname = '{args.db_name}'")
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(args.db_name)))
            print(f"[+] Created database: {args.db_name}")
        else:
            print(f"[*] Database already exists: {args.db_name}")
        cur.close()
        admin_conn.close()

    # ── Create schema ──────────────────────────────────────────────────
    conn = _connect_with_retry(conn_params)
    cur = conn.cursor()
    cur.execute(CREATE_TABLES_SQL)
    conn.commit()
    cur.close()
    conn.close()
    print("[+] Schema ready")

    # ── Resolve all file paths ─────────────────────────────────────────
    input_dir = args.input
    all_tsv_paths = []
    for _day, rel_path in ATTACK_DAY_MAP:
        p = os.path.join(input_dir, rel_path)
        if not os.path.exists(p):
            print(f"[!] Missing file: {p}")
            sys.exit(1)
        all_tsv_paths.append(p)
    benign_path = os.path.join(input_dir, DAY1_BENIGN_SOURCE)
    if not os.path.exists(benign_path):
        print(f"[!] Missing benign source: {benign_path}")
        sys.exit(1)
    print(f"[*] All 43 TSV files found")

    # ── Phase 1: IP index (cached) ────────────────────────────────────
    ip_to_index, idx_path = ensure_ip_index(input_dir, all_tsv_paths)

    # ── Phase 2: netflow nodes ─────────────────────────────────────────
    insert_netflow_nodes(conn_params, ip_to_index)

    # ── Compute event_counter from existing max _id ────────────────────
    resume_day = args.resume_from_day
    event_counter = 0
    if resume_day > 0:
        conn = _connect_with_retry(conn_params)
        cur = conn.cursor()
        cur.execute("SELECT COALESCE(MAX(_id), 0) FROM event_table")
        event_counter = cur.fetchone()[0]
        cur.close()
        conn.close()
        print(f"[*] Resuming from day {resume_day}, "
              f"event_counter starts at {event_counter}")

    total_events = 0
    grand_t0 = time.time()

    # ── Phase 3: Day 1 — benign only from single file ─────────────────
    if resume_day <= 1:
        print(f"[*] Phase 3: Day 1 — benign from "
              f"{os.path.basename(DAY1_BENIGN_SOURCE)}...", end="", flush=True)
        n, elapsed = _ingest_one_day(
            conn_params, idx_path, benign_path,
            day=1, event_offset=event_counter, benign_only=True,
        )
        event_counter += n
        total_events += n
        print(f" {n:,} events ({elapsed:.1f}s)")
    else:
        print(f"[*] Phase 3: Skipping Day 1 (resume_from_day={resume_day})")

    # ── Phase 4: Days 2-44 — full rows (benign + attack) ──────────────
    print("[*] Phase 4: Ingesting attack days (2-44)...")
    for day, rel_path in ATTACK_DAY_MAP:
        if day < resume_day:
            continue

        path = os.path.join(input_dir, rel_path)
        attack_name = os.path.splitext(os.path.basename(rel_path))[0]
        role = "val " if day <= 8 else "test"
        print(f"    Day {day:2d} [{role}] {attack_name}...", end="", flush=True)

        n, elapsed = _ingest_one_day(
            conn_params, idx_path, path,
            day=day, event_offset=event_counter, benign_only=False,
        )
        event_counter += n
        total_events += n
        print(f" {n:,} events ({elapsed:.1f}s)")

    # ── Create index ───────────────────────────────────────────────────
    print("[*] Creating index on event_table.timestamp_rec...")
    conn = _connect_with_retry(conn_params)
    cur = conn.cursor()
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_event_ts "
        "ON event_table (timestamp_rec);")
    conn.commit()
    cur.close()
    conn.close()

    grand_elapsed = time.time() - grand_t0
    print(f"[+] Total events inserted this run: {total_events:,}")
    print(f"[+] Total event_counter: {event_counter:,}")
    print(f"[+] Elapsed: {grand_elapsed/60:.1f} min")
    print("[+] Done")


if __name__ == "__main__":
    main()
