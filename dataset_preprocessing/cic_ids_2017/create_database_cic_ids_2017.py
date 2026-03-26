#!/usr/bin/env python3
"""
CIC-IDS-2017 Dataset Ingestion Script for PIDSMaker

Reads CIC-IDS-2017 CSV files (network flow data) and inserts them into
PostgreSQL using PIDSMaker's existing schema (event_table, netflow_node_table).

Since CIC-IDS-2017 is purely network-level data (IP addresses only),
subject_node_table and file_node_table remain empty.

Usage:
    python3 create_database_cic_ids_2017.py \
        --input /path/to/CIC-IDS-2017/ \
        --db-name cic_ids_2017 \
        --host localhost --port 5432 \
        --user postgres --password postgres
"""

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2 import sql
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
except ImportError:
    print("[!] Missing psycopg2. Install with: pip install psycopg2-binary")
    sys.exit(1)


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

CSV_FILES = {
    "Monday-WorkingHours.csv": 3,
    "Tuesday-WorkingHours.csv": 4,
    "Wednesday-WorkingHours.csv": 5,
    "Thursday-WorkingHours.csv": 6,
    "Friday-WorkingHours.csv": 7,
}

VALID_EDGE_TYPES = {"TCP", "UDP", "Other"}


def sha256_hash(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# Worker script executed in a subprocess that never imports psycopg2.
# Python 3.13 + psycopg2 C-extension causes memory corruption during large
# binary file iteration. A clean subprocess avoids this entirely.
_WORKER_SCRIPT = r"""
import hashlib, json, sys
with open(sys.argv[1]) as jf:
    ip_to_index = json.load(jf)
csv_path   = sys.argv[2]
out_path   = sys.argv[3]
evt_offset = int(sys.argv[4])
VALID = {"TCP", "UDP", "Other"}
n = 0
with open(out_path, "wb") as out, open(csv_path, "rb") as f:
    next(f)
    for raw in f:
        p = raw.replace(b"\x00", b"").split(b",")
        if len(p) < 6:
            continue
        src_ip = p[0].decode("utf-8", "replace").strip()
        dst_ip = p[2].decode("utf-8", "replace").strip()
        op     = p[4].decode("utf-8", "replace").strip()
        ts     = p[5].decode("utf-8", "replace").strip()
        if op not in VALID:
            continue
        si = ip_to_index.get(src_ip)
        di = ip_to_index.get(dst_ip)
        if si is None or di is None:
            continue
        try:
            ts_ns = int(float(ts) * 1_000_000_000)
        except Exception:
            continue
        sh = hashlib.sha256(src_ip.encode()).hexdigest()
        dh = hashlib.sha256(dst_ip.encode()).hexdigest()
        line = f"{sh}\t{si}\t{op}\t{dh}\t{di}\tevt_{evt_offset + n}\t{ts_ns}\n"
        out.write(line.encode("utf-8"))
        n += 1
print(n)
"""


class CICIDSIngestor:
    def __init__(self, db_name, host, port, user, password):
        self.db_name = db_name
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.conn = None
        self.cur = None

    def create_database(self):
        conn = psycopg2.connect(
            dbname="postgres", user=self.user, password=self.password,
            host=self.host, port=self.port,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.db_name}'")
        if not cur.fetchone():
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.db_name)))
            print(f"[+] Created database: {self.db_name}")
        else:
            print(f"[*] Database already exists: {self.db_name}")
        cur.close()
        conn.close()

    def connect(self):
        self.conn = psycopg2.connect(
            dbname=self.db_name, user=self.user, password=self.password,
            host=self.host, port=self.port,
        )
        self.cur = self.conn.cursor()
        print(f"[+] Connected to database: {self.db_name}")

    def create_schema(self):
        self.cur.execute(CREATE_TABLES_SQL)
        self.conn.commit()
        print("[+] Created table schema")

    def ingest(self, input_dir):
        print("[*] Truncating existing tables...")
        self.cur.execute(
            "TRUNCATE TABLE event_table, netflow_node_table, "
            "subject_node_table, file_node_table RESTART IDENTITY;"
        )
        self.conn.commit()

        # Phase 1: collect unique IPs
        # Close the psycopg2 connection before large-scale file I/O.
        # Python 3.13's memory allocator corrupts memory when psycopg2's C extension
        # is active (connected) during 46M+ row binary file iteration, even with gc.disable().
        print("[*] Phase 1: Scanning for unique IP addresses...")
        self.cur.close()
        self.conn.close()
        self.cur = None
        self.conn = None

        ip_set = set()
        file_paths = {}
        for csv_file, day in CSV_FILES.items():
            path = os.path.join(input_dir, csv_file)
            if not os.path.exists(path):
                print(f"[!] Missing file: {path}")
                continue
            file_paths[csv_file] = path
            with open(path, "rb") as f:
                next(f)
                for raw_line in f:
                    parts = raw_line.replace(b"\x00", b"").split(b",")
                    if len(parts) >= 3:
                        ip_set.add(parts[0].decode("utf-8", errors="replace").strip())
                        ip_set.add(parts[2].decode("utf-8", errors="replace").strip())
        ip_set.discard("")
        print(f"    Found {len(ip_set)} unique IP addresses")

        # Reconnect for Phase 2
        self.connect()

        # Phase 2: insert netflow nodes
        print("[*] Phase 2: Creating netflow nodes...")
        ip_to_index = {}
        for idx, ip in enumerate(sorted(ip_set)):
            ip_to_index[ip] = idx
            self.cur.execute(
                """INSERT INTO netflow_node_table
                   (node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port, index_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (ip, sha256_hash(ip), ip, "0", ip, "0", idx),
            )
        self.conn.commit()
        print(f"    Inserted {len(ip_to_index)} netflow nodes")

        # Phase 3: ingest events via subprocess worker + COPY FROM temp file
        # Spawning a fresh Python process (no psycopg2) to process each CSV into a
        # temp TSV fully isolates large file I/O from psycopg2's C extension.
        print("[*] Phase 3: Ingesting events...")
        total_events = 0
        event_counter = 0

        # Write ip_to_index to a temp file — passing 400KB of JSON as argv exceeds ARG_MAX
        idx_fd, idx_path = tempfile.mkstemp(suffix=".json", prefix="cicids_idx_")
        with os.fdopen(idx_fd, "w") as jf:
            json.dump(ip_to_index, jf)

        try:
            for csv_file, path in file_paths.items():
                day = CSV_FILES[csv_file]
                print(f"    Processing {csv_file} (day {day})...", flush=True)

                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tsv", prefix="cicids_")
                os.close(tmp_fd)
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", _WORKER_SCRIPT,
                         idx_path, path, tmp_path, str(event_counter)],
                        capture_output=True, text=True,
                    )
                    if result.returncode != 0:
                        print(f"[!] Worker error for {csv_file} (rc={result.returncode}):\n"
                              f"  stderr: {result.stderr[:400]}\n"
                              f"  stdout: {result.stdout[:100]}")
                        continue

                    n = int(result.stdout.strip())
                    event_counter += n

                    with open(tmp_path, "rb") as f:
                        self.cur.copy_expert(
                            "COPY event_table (src_node, src_index_id, operation, "
                            "dst_node, dst_index_id, event_uuid, timestamp_rec) FROM STDIN",
                            f,
                        )
                    self.conn.commit()
                    total_events += n
                    print(f"      -> {n:,} events inserted")

                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        finally:
            if os.path.exists(idx_path):
                os.unlink(idx_path)

        print(f"[+] Total events inserted: {total_events:,}")

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print("[+] Database connection closed")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest CIC-IDS-2017 into PostgreSQL for PIDSMaker"
    )
    parser.add_argument("--input", required=True,
                        help="Path to CIC-IDS-2017 directory containing CSV files")
    parser.add_argument("--db-name", default="cic_ids_2017")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--create-db", action="store_true",
                        help="Create database if it doesn't exist")
    args = parser.parse_args()

    ingestor = CICIDSIngestor(
        db_name=args.db_name, host=args.host, port=args.port,
        user=args.user, password=args.password,
    )
    if args.create_db:
        ingestor.create_database()
    ingestor.connect()
    ingestor.create_schema()
    ingestor.ingest(args.input)
    ingestor.close()


if __name__ == "__main__":
    main()
