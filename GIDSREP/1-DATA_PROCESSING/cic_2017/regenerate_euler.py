#!/usr/bin/env python3
"""
Fast euler/ regeneration from existing cic_2017_1.csv.
Fixes the port-number bug: uses Source IP / Destination IP instead of
Source Port / Destination Port as node identifiers.

Usage:
    python regenerate_euler.py            # regenerate euler/ only
    python regenerate_euler.py --dst ./euler_new/   # custom output dir
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

ap = argparse.ArgumentParser(description='Regenerate euler/ data from cic_2017_1.csv')
ap.add_argument('--src', default='cic_2017_1.csv', help='Merged CIC-IDS CSV')
ap.add_argument('--dst', default='./euler/', help='Output directory')
ap.add_argument('--delta', type=int, default=100000, help='File split delta (timestamp units)')
args = ap.parse_args()

DELTA = args.delta
DST = args.dst.rstrip('/') + '/'
SRC = args.src

assert os.path.exists(SRC), f'{SRC} not found. Run process_cic_2017.py first.'

print(f'Reading {SRC} ...')
t0 = time.time()
result = pd.read_csv(SRC, header=0, index_col=0)
result = result.replace(result.iloc[97]['Flow Bytes/s'], 0)
result = result.replace(np.nan, 0)

# Create binary label
la = result[' Label'].values
result['label'] = [0 if i == 'BENIGN' else 1 for i in la]
print(f'  Loaded {len(result)} rows in {time.time()-t0:.1f}s')

# Drop unneeded columns (same as process_cic_2017.py)
result_argus = result.copy()
result_argus.drop(columns=['Flow ID', ' Label', ' Timestamp'], inplace=True)

# Select columns by NAME — the actual fix for the port-number bug
euler_cols = result_argus[['time_int', ' Source IP', ' Destination IP',
                           ' Flow Duration', 'Total Length of Fwd Packets',
                           ' Total Fwd Packets', 'label']]
data1_euler = euler_cols.values

os.makedirs(DST, exist_ok=True)

# Node map: string node ID → integer
nmap = {}
nid = [0]

def get_or_add(n):
    if n not in nmap:
        nmap[n] = nid[0]
        nid[0] += 1
    return nmap[n]

fmt_line = lambda ts, src, dst, dur, bytes_, pkts, lbl: (
    '%s,%s,%s,%s,%s,%s,%s\n' % (
        int(ts), get_or_add(src), get_or_add(dst), dur, bytes_, pkts, lbl
    ),
    int(ts)
)

print(f'Generating {DST} ...')
t0 = time.time()

cur_time = 0
f_out = open(DST + str(cur_time) + '.txt', 'w+')

for tokens in data1_euler:
    # [0]=time_int, [1]=Source IP, [2]=Destination IP,
    # [3]=Flow Duration, [4]=Total Length of Fwd Packets,
    # [5]=Total Fwd Packets, [6]=label
    l, ts = fmt_line(tokens[0], tokens[1], tokens[2], tokens[3],
                     tokens[4], tokens[5], tokens[6])
    if ts >= cur_time + DELTA:
        cur_time += DELTA
        f_out.close()
        f_out = open(DST + str(cur_time) + '.txt', 'w+')
    f_out.write(l)
f_out.close()

# Build reverse node map and save
nmap_rev = [None] * (max(nmap.values()) + 1)
for (k, v) in nmap.items():
    nmap_rev[v] = k
with open(DST + 'nmap.pkl', 'wb+') as f:
    pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)

elapsed = time.time() - t0
n_files = len([f for f in os.listdir(DST) if f.endswith('.txt')])

print(f'\nDone in {elapsed:.1f}s')
print(f'  Output dir : {DST}')
print(f'  Files      : {n_files} .txt + nmap.pkl')
print(f'  Nodes      : {len(nmap_rev)}')
print(f'  Node type  : {type(nmap_rev[0]).__name__} (should be "str" = IP addresses)')
print(f'  Sample IPs : {nmap_rev[:5]}')
