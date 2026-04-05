#!/usr/bin/env python3
"""
Export THEIA_E3 and THEIA_E5 datasets from PostgreSQL to gzipped CSV files.

This script connects to the PostgreSQL database and exports:
- Node tables (netflow, subject/process, file)
- Edge table

All files are directly written as .csv.gz (gzip compressed) for efficiency.

Usage:
    1. Start the database first:
       cd /scratch/asawan15/PIDSMaker/scripts/apptainer && make up
    
    2. Run this script:
       python scripts/export_theia_datasets.py

Output:
    Creates a folder: /scratch/asawan15/PIDSMaker/theia_csv_exports/
    With subfolders for each dataset containing gzipped CSV files.
"""

import os
import csv
import gzip
import psycopg2
from datetime import datetime

# Database connection settings
DB_HOST = "localhost"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_PORT = 5432

# Output directory
BASE_DIR = "/scratch/asawan15/PIDSMaker"
OUTPUT_DIR = os.path.join(BASE_DIR, "theia_csv_exports")

# Datasets to export
DATASETS = {
    "THEIA_E3": "theia_e3",
    "THEIA_E5": "theia_e5",
}


def connect_to_database(database_name):
    """Connect to PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            database=database_name,
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
        )
        return conn
    except psycopg2.Error as e:
        print(f"  ERROR: Could not connect to database '{database_name}': {e}")
        return None


def export_table_to_csv(cursor, table_name, output_path, columns=None, chunk_size=500000):
    """Export a database table to gzipped CSV in chunks to avoid memory issues."""
    try:
        # First get column names
        if columns:
            sql = f"SELECT {', '.join(columns)} FROM {table_name} LIMIT 0;"
        else:
            sql = f"SELECT * FROM {table_name} LIMIT 0;"
        
        cursor.execute(sql)
        col_names = [desc[0] for desc in cursor.description]
        
        # Get total count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        total_rows = cursor.fetchone()[0]
        
        # Open gzip file and write header
        with gzip.open(output_path, 'wt', newline='', encoding='utf-8', compresslevel=6) as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            
            # Export in chunks using OFFSET/LIMIT
            offset = 0
            rows_written = 0
            
            while offset < total_rows:
                if columns:
                    sql = f"SELECT {', '.join(columns)} FROM {table_name} ORDER BY 1 LIMIT {chunk_size} OFFSET {offset};"
                else:
                    sql = f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset};"
                
                cursor.execute(sql)
                records = cursor.fetchall()
                
                if not records:
                    break
                
                writer.writerows(records)
                rows_written += len(records)
                offset += chunk_size
                
                # Progress indicator
                if total_rows > chunk_size:
                    pct = min(100, (rows_written / total_rows) * 100)
                    print(f"\r  Exporting {table_name}... {rows_written:,}/{total_rows:,} ({pct:.1f}%)", end="", flush=True)
            
            if total_rows > chunk_size:
                print()  # New line after progress
        
        return rows_written
    except (psycopg2.Error, Exception) as e:
        print(f"\n  ERROR exporting {table_name}: {e}")
        return 0


def export_dataset(dataset_name, database_name):
    """Export all tables from a dataset to CSV files."""
    print(f"\n{'='*60}")
    print(f"Exporting {dataset_name} (database: {database_name})")
    print(f"{'='*60}")
    
    # Create output directory
    out_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Connect to database
    conn = connect_to_database(database_name)
    if conn is None:
        return False
    
    cursor = conn.cursor()
    
    # Tables to export (now with .gz extension)
    tables = {
        "netflow_node_table": "netflow_nodes.csv.gz",
        "subject_node_table": "subject_nodes.csv.gz", 
        "file_node_table": "file_nodes.csv.gz",
        "event_table": "edges.csv.gz",
    }
    
    summary = {}
    
    for table_name, csv_filename in tables.items():
        output_path = os.path.join(out_dir, csv_filename)
        print(f"  Exporting {table_name}...", end=" " if table_name != "event_table" else "\n", flush=True)
        
        count = export_table_to_csv(cursor, table_name, output_path)
        summary[table_name] = count
        if table_name != "event_table":
            print(f"{count:,} rows")
    
    # Write summary
    summary_path = os.path.join(out_dir, "export_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Dataset Export Summary: {dataset_name}\n")
        f.write(f"Database: {database_name}\n")
        f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")
        
        total_rows = 0
        for table_name, count in summary.items():
            f.write(f"{table_name}: {count:,} rows\n")
            total_rows += count
        
        f.write(f"\nTotal rows exported: {total_rows:,}\n")
    
    cursor.close()
    conn.close()
    
    print(f"  Summary written to: {summary_path}")
    
    # Print file sizes
    print(f"  File sizes:")
    for csv_filename in tables.values():
        fpath = os.path.join(out_dir, csv_filename)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"    {csv_filename}: {size_mb:.2f} MB")
    
    return True


def main():
    print("=" * 60)
    print("THEIA Dataset CSV Exporter (gzip compressed)")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Export each dataset
    successful_exports = []
    for dataset_name, database_name in DATASETS.items():
        success = export_dataset(dataset_name, database_name)
        if success:
            successful_exports.append(dataset_name)
    
    if not successful_exports:
        print("\nERROR: No datasets were exported successfully!")
        print("Make sure the database is running:")
        print("  cd /scratch/asawan15/PIDSMaker/scripts/apptainer && make up")
        return 1
    
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"Exported datasets: {', '.join(successful_exports)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nAll files are gzip compressed (.csv.gz)")
    print("To read in Python: pd.read_csv('file.csv.gz', compression='gzip')")
    
    return 0


if __name__ == "__main__":
    exit(main())
