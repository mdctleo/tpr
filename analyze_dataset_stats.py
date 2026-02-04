#!/usr/bin/env python3
"""
Script to analyze dataset statistics from PostgreSQL .dump files in the data folder.

Computes:
- Total number of unique edges
- Number of edges with timestamps > 10
- Mean/median number of timestamps for edges with timestamps > 10
"""

import os
import subprocess
import tempfile
from collections import defaultdict
import numpy as np
import psycopg2


def restore_dump_to_temp_db(dump_file, db_name):
    """Restore a PostgreSQL dump file to a temporary database."""
    # Create the database
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', os.environ.get('USER', 'postgres'))
    db_password = os.environ.get('DB_PASSWORD', 'postgres')
    
    conn = psycopg2.connect(
        dbname='postgres', 
        user=db_user,
        password=db_password,
        host=db_host
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    # Drop if exists and create new
    cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
    cur.execute(f"CREATE DATABASE {db_name}")
    cur.close()
    conn.close()
    
    # Restore the dump
    cmd = f"pg_restore -d {db_name} {dump_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0 and "already exists" not in result.stderr:
        print(f"Warning during restore: {result.stderr}")
    
    return db_name


def analyze_database(db_name):
    """
    Analyze edge statistics from the PostgreSQL database using SQL aggregation.
    
    Returns:
        dict: Statistics including unique edges, edges with >10 timestamps, mean/median
    """
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', os.environ.get('USER', 'postgres'))
    db_password = os.environ.get('DB_PASSWORD', 'postgres')
    
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host
        )
    except Exception as e:
        print(f"Could not connect to database {db_name}: {e}")
        return None
    
    cur = conn.cursor()
    
    try:
        # First, get total unique edges
        print("  Counting total unique edges...")
        query_total = """
            SELECT COUNT(DISTINCT (src_node, dst_node))
            FROM event_table
        """
        cur.execute(query_total)
        total_unique_edges = cur.fetchone()[0]
        
        # Get edge statistics using SQL aggregation for edges with > 10 timestamps
        print("  Analyzing edge timestamp frequencies (> 10 timestamps)...")
        query_stats = """
            SELECT 
                src_node,
                dst_node,
                COUNT(*) as timestamp_count
            FROM event_table
            GROUP BY src_node, dst_node
            HAVING COUNT(*) > 10
        """
        cur.execute(query_stats)
        
        # Fetch all results
        results = cur.fetchall()
        num_edges_gt_10 = len(results)
        
        if num_edges_gt_10 > 0:
            timestamp_counts = [row[2] for row in results]
            mean_timestamps = np.mean(timestamp_counts)
            median_timestamps = np.median(timestamp_counts)
            min_timestamps = np.min(timestamp_counts)
            max_timestamps = np.max(timestamp_counts)
            std_timestamps = np.std(timestamp_counts)
            
            # Additional percentile statistics
            percentile_25 = np.percentile(timestamp_counts, 25)
            percentile_75 = np.percentile(timestamp_counts, 75)
            percentile_90 = np.percentile(timestamp_counts, 90)
            percentile_95 = np.percentile(timestamp_counts, 95)
            percentile_99 = np.percentile(timestamp_counts, 99)
        else:
            timestamp_counts = []
            mean_timestamps = 0
            median_timestamps = 0
            min_timestamps = 0
            max_timestamps = 0
            std_timestamps = 0
            percentile_25 = 0
            percentile_75 = 0
            percentile_90 = 0
            percentile_95 = 0
            percentile_99 = 0
        
        cur.close()
        conn.close()
        
        return {
            'total_unique_edges': total_unique_edges,
            'num_edges_gt_10': num_edges_gt_10,
            'mean_timestamps_gt_10': mean_timestamps,
            'median_timestamps_gt_10': median_timestamps,
            'min_timestamps_gt_10': min_timestamps,
            'max_timestamps_gt_10': max_timestamps,
            'std_timestamps_gt_10': std_timestamps,
            'percentile_25_gt_10': percentile_25,
            'percentile_75_gt_10': percentile_75,
            'percentile_90_gt_10': percentile_90,
            'percentile_95_gt_10': percentile_95,
            'percentile_99_gt_10': percentile_99,
            'timestamp_counts_gt_10': timestamp_counts
        }
        
    except Exception as e:
        print(f"Error querying database: {e}")
        cur.close()
        conn.close()
        return None


def cleanup_database(db_name):
    """Drop the temporary database."""
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', os.environ.get('USER', 'postgres'))
    db_password = os.environ.get('DB_PASSWORD', 'postgres')
    
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user=db_user,
            password=db_password,
            host=db_host
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not cleanup database {db_name}: {e}")


def check_database_exists(db_name):
    """Check if a database already exists."""
    db_host = os.environ.get('DB_HOST', 'localhost')
    db_user = os.environ.get('DB_USER', os.environ.get('USER', 'postgres'))
    db_password = os.environ.get('DB_PASSWORD', 'postgres')
    
    try:
        conn = psycopg2.connect(
            dbname='postgres',
            user=db_user,
            password=db_password,
            host=db_host
        )
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone() is not None
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        print(f"Could not check database existence: {e}")
        return False


def main():
    """Main function to analyze all datasets in the data folder."""
    # Use relative path that works both inside and outside Docker
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # Find all .dump files
    dump_files = [f for f in os.listdir(data_dir) if f.endswith('.dump')]
    
    if not dump_files:
        print(f"No .dump files found in {data_dir}")
        return
    
    print("=" * 80)
    print("DATASET STATISTICS ANALYSIS")
    print("=" * 80)
    print()
    
    for dump_file in sorted(dump_files):
        filepath = os.path.join(data_dir, dump_file)
        dataset_name = dump_file.replace('.dump', '')
        
        # Try both the dataset name directly and with temp prefix
        possible_db_names = [dataset_name, f"temp_analysis_{dataset_name}"]
        
        print(f"Analyzing: {dataset_name}")
        print("-" * 80)
        
        db_name = None
        needs_cleanup = False
        
        try:
            # Check if database already exists
            for name in possible_db_names:
                if check_database_exists(name):
                    print(f"Found existing database: {name}")
                    db_name = name
                    break
            
            # If not found, restore from dump
            if db_name is None:
                db_name = f"temp_analysis_{dataset_name}"
                print(f"Restoring {dump_file} to temporary database {db_name}...")
                restore_dump_to_temp_db(filepath, db_name)
                needs_cleanup = True
            
            # Analyze the database
            print(f"Analyzing edge statistics...")
            stats = analyze_database(db_name)
            
            if stats:
                print()
                print(f"{'='*60}")
                print(f"OVERALL STATISTICS")
                print(f"{'='*60}")
                print(f"Total unique edges: {stats['total_unique_edges']:,}")
                print(f"Edges with > 10 timestamps: {stats['num_edges_gt_10']:,}")
                print(f"Percentage with > 10 timestamps: {stats['num_edges_gt_10'] / stats['total_unique_edges'] * 100:.2f}%")
                print()
                print(f"{'='*60}")
                print(f"STATISTICS FOR EDGES WITH > 10 TIMESTAMPS")
                print(f"{'='*60}")
                print(f"Mean timestamps: {stats['mean_timestamps_gt_10']:.2f}")
                print(f"Median timestamps: {stats['median_timestamps_gt_10']:.2f}")
                print(f"Min timestamps: {stats['min_timestamps_gt_10']}")
                print(f"Max timestamps: {stats['max_timestamps_gt_10']}")
                print(f"Std dev timestamps: {stats['std_timestamps_gt_10']:.2f}")
                print()
                print(f"PERCENTILES:")
                print(f"  25th percentile: {stats['percentile_25_gt_10']:.2f}")
                print(f"  75th percentile: {stats['percentile_75_gt_10']:.2f}")
                print(f"  90th percentile: {stats['percentile_90_gt_10']:.2f}")
                print(f"  95th percentile: {stats['percentile_95_gt_10']:.2f}")
                print(f"  99th percentile: {stats['percentile_99_gt_10']:.2f}")
            else:
                print("Failed to analyze dataset")
                
        except Exception as e:
            print(f"Error analyzing {dump_file}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup temporary database only if we created it
            if needs_cleanup and db_name:
                print(f"Cleaning up temporary database...")
                cleanup_database(db_name)
        
        print()
        print()


if __name__ == '__main__':
    main()
