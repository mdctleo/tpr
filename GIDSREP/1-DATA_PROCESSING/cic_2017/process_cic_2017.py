#!/usr/bin/env python3
"""
Standalone processing script derived from process_cic_2017.ipynb
Processes the CIC-IDS 2017 dataset into formats needed by each model:
  1. cic_2017_1.csv - merged, cleaned, labeled dataset (for Anomal-E)
  2. cic_argus.csv + argus_flow/ directory - for ARGUS (flow variant)
  3. euler/ directory - for EULER and VGRNN
  4. cic_20.csv - for PIKACHU (20-minute snapshots)
"""

import pandas as pd
from datetime import datetime
import numpy as np
import pickle
import time
import csv
import os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORK_DIR)

print("=" * 60)
print("CIC-IDS 2017 Data Processing Pipeline")
print("=" * 60)

# =============================================================================
# STEP 1: Load Monday
# =============================================================================
print("\n[1/8] Loading Monday-WorkingHours...")
# Use latin-1 encoding to handle non-UTF8 bytes in CIC-IDS CSVs
READ_OPTS = dict(encoding='latin-1', low_memory=False)
mon = pd.read_csv('Monday-WorkingHours.pcap_ISCX.csv', **READ_OPTS)
time1 = mon[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M:%S')
    if tmp.hour < 8:
        time2.append(int(tmp.timestamp()) + 43200)
    else:
        time2.append(int(tmp.timestamp()))
mon['time_int'] = time2
mon1 = mon.sort_values(by='time_int')
print(f"  Monday rows: {len(mon1)}")

# =============================================================================
# STEP 2: Load Tuesday
# =============================================================================
print("[2/8] Loading Tuesday-WorkingHours...")
tue = pd.read_csv('Tuesday-WorkingHours.pcap_ISCX.csv', **READ_OPTS)
time1 = tue[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    if tmp.hour < 8:
        time2.append(int(tmp.timestamp()) + 43200)
    else:
        time2.append(int(tmp.timestamp()))
tue['time_int'] = time2
tue1 = tue.sort_values(by='time_int')
print(f"  Tuesday rows: {len(tue1)}")

# =============================================================================
# STEP 3: Load Wednesday
# =============================================================================
print("[3/8] Loading Wednesday-workingHours...")
wed = pd.read_csv('Wednesday-workingHours.pcap_ISCX.csv', **READ_OPTS)
time1 = wed[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    if tmp.hour < 8:
        time2.append(int(tmp.timestamp()) + 43200)
    else:
        time2.append(int(tmp.timestamp()))
wed['time_int'] = time2
wed1 = wed.sort_values(by='time_int')
print(f"  Wednesday rows: {len(wed1)}")

# =============================================================================
# STEP 4-1: Load Thursday Morning (WebAttacks)
# =============================================================================
print("[4/8] Loading Thursday-Morning-WebAttacks...")
Thu_1 = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', **READ_OPTS)
Thu_1 = Thu_1.iloc[:170365]
time1 = Thu_1[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    time2.append(int(tmp.timestamp()))
Thu_1['time_int'] = time2
Thu_11 = Thu_1.sort_values(by='time_int')
print(f"  Thursday AM rows: {len(Thu_11)}")

# =============================================================================
# STEP 4-2: Load Thursday Afternoon (Infilteration)
# =============================================================================
print("[5/8] Loading Thursday-Afternoon-Infilteration...")
Thu_2 = pd.read_csv('Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', **READ_OPTS)
time1 = Thu_2[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    time2.append(int(tmp.timestamp()) + 43200)
Thu_2['time_int'] = time2
Thu_21 = Thu_2.sort_values(by='time_int')
print(f"  Thursday PM rows: {len(Thu_21)}")

# =============================================================================
# STEP 5-1: Load Friday Morning
# =============================================================================
print("[6/8] Loading Friday-Morning...")
Fri_1 = pd.read_csv('Friday-WorkingHours-Morning.pcap_ISCX.csv', **READ_OPTS)
time1 = Fri_1[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    time2.append(int(tmp.timestamp()))
Fri_1['time_int'] = time2
Fri_11 = Fri_1.sort_values(by='time_int')
print(f"  Friday AM rows: {len(Fri_11)}")

# =============================================================================
# STEP 5-2: Load Friday Afternoon (PortScan)
# =============================================================================
print("[7/8] Loading Friday-Afternoon-PortScan...")
Fri_2 = pd.read_csv('Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', **READ_OPTS)
time1 = Fri_2[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    time2.append(int(tmp.timestamp()) + 43200)
Fri_2['time_int'] = time2
Fri_21 = Fri_2.sort_values(by='time_int')
print(f"  Friday PM PortScan rows: {len(Fri_21)}")

# =============================================================================
# STEP 5-3: Load Friday Afternoon (DDoS)
# =============================================================================
print("[8/8] Loading Friday-Afternoon-DDos...")
Fri_3 = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', **READ_OPTS)
time1 = Fri_3[' Timestamp'].values
time2 = []
for i in time1:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M')
    time2.append(int(tmp.timestamp()) + 43200)
Fri_3['time_int'] = time2
Fri_31 = Fri_3.sort_values(by='time_int')
print(f"  Friday PM DDoS rows: {len(Fri_31)}")

# =============================================================================
# MERGE ALL
# =============================================================================
print("\n" + "=" * 60)
print("Merging all days...")
frame = [mon1, tue1, wed1, Thu_11, Thu_21, Fri_11, Fri_21, Fri_31]
result = pd.concat(frame)
result = result.reset_index(drop=True)
result = result.sort_values(by='time_int')
result = result.reset_index(drop=True)
result['time_int'] = result['time_int'] - result['time_int'].iloc[0]

# Create binary label
la = result[' Label'].values
label = []
for i in la:
    if i != 'BENIGN':
        label.append(1)
    else:
        label.append(0)
result['label'] = label

attack_count = sum(label)
total_count = len(label)
print(f"  Total rows: {total_count}")
print(f"  Attack rows: {attack_count}")
print(f"  Benign rows: {total_count - attack_count}")

# =============================================================================
# OUTPUT 1: Anomal-E format (cic_2017_1.csv)
# =============================================================================
print("\n" + "=" * 60)
print("Saving cic_2017_1.csv (Anomal-E format)...")
result.to_csv('cic_2017_1.csv', encoding='utf-8')
print("  Done.")

# Reload and clean for further processing
result = pd.read_csv('cic_2017_1.csv', header=0, index_col=0)
result = result.replace(result.iloc[97]['Flow Bytes/s'], 0)
result = result.replace(np.nan, 0)

# =============================================================================
# OUTPUT 2: ARGUS flow format (cic_argus.csv + argus_flow/)
# =============================================================================
print("\n" + "=" * 60)
print("Generating ARGUS flow format...")

# Drop unneeded columns for ARGUS
result_argus = result.copy()
result_argus.drop(columns=['Flow ID', ' Label', ' Timestamp'], inplace=True)

data = result_argus[['time_int', ' Source IP', ' Destination IP', 'label',
                      ' Flow Duration', ' Total Fwd Packets',
                      ' Total Backward Packets',
                      'Total Length of Fwd Packets',
                      ' Total Length of Bwd Packets']]

print("  Writing cic_argus.csv...")
start_time = time.time()
with open('cic_argus.csv', 'w+', newline='') as output:
    csv_writer = csv.writer(output)
    csv_writer.writerow(['timestamp', 'src_computer', 'dst_computer',
                         'dur', 'bytes', 'pkts', 'label'])
    hh = data.values
    num = 0
    for flow in hh:
        num += 1
        if num % 1000000 == 0:
            print(f"    {num} rows, {time.time()-start_time:.1f}s")
        ts = flow[0]
        src = flow[1]
        dst = flow[2]
        dur = 0 if flow[4] == '-' else flow[4]
        ori_byte = 0 if flow[7] == '-' else int(flow[7])
        resp_bytes = 0 if flow[8] == '-' else int(flow[8])
        orig_pkts = 0 if flow[5] == '-' else int(flow[5])
        resp_pkts = 0 if flow[6] == '-' else int(flow[6])
        lbl = flow[3]
        csv_writer.writerow([str(num), ts, src, dst, dur,
                             resp_bytes + ori_byte,
                             orig_pkts + resp_pkts, lbl])
print(f"  cic_argus.csv done in {time.time()-start_time:.1f}s")

# Now generate the argus_flow/ directory
print("  Writing argus_flow/ directory...")
data_argus = pd.read_csv('cic_argus.csv', header=0)
DELTA = 100000
DST = './argus_flow/'
os.makedirs(DST, exist_ok=True)

data1 = data_argus.values
min_val = data1[0][0]
time_set = np.array((data1[data_argus[data_argus['label'] == 1].index[0] - 1][0],
                      data1[-1][0]), dtype=np.int32)
print(f"  time_set: {time_set}")

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

cur_time = 0
f_out = open(DST + str(cur_time) + '.txt', 'w+')
for tokens in data1:
    l, ts = fmt_line(tokens[0], tokens[1], tokens[2], tokens[3],
                     tokens[4], tokens[5], tokens[6])
    if ts >= cur_time + DELTA:
        cur_time += DELTA
        f_out.close()
        f_out = open(DST + str(cur_time) + '.txt', 'w+')
    f_out.write(l)
f_out.close()

nmap_rev = [None] * (max(nmap.values()) + 1)
for (k, v) in nmap.items():
    nmap_rev[v] = k
with open(DST + 'nmap.pkl', 'wb+') as f:
    pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)

data_inf = np.array((len(nmap_rev),
                      data_argus[data_argus['label'] == 1].index[0]),
                     dtype=np.int32)
print(f"  argus_flow/ data_inf: {data_inf}")

# =============================================================================
# OUTPUT 3: EULER & VGRNN format (euler/)
# =============================================================================
print("\n" + "=" * 60)
print("Generating EULER/VGRNN format...")

# Select columns by NAME to avoid column-index bugs (Source IP, not Source Port)
euler_cols = result_argus[['time_int', ' Source IP', ' Destination IP',
                           ' Flow Duration', 'Total Length of Fwd Packets',
                           ' Total Fwd Packets', 'label']]
data1_euler = euler_cols.values
DELTA = 100000
DST = './euler/'
os.makedirs(DST, exist_ok=True)

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

cur_time = 0
f_out = open(DST + str(cur_time) + '.txt', 'w+')

for tokens in data1_euler:
    # Columns selected by name above:
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

nmap_rev = [None] * (max(nmap.values()) + 1)
for (k, v) in nmap.items():
    nmap_rev[v] = k
with open(DST + 'nmap.pkl', 'wb+') as f:
    pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"  euler/ done. Nodes: {len(nmap_rev)}")

# =============================================================================
# OUTPUT 4: PIKACHU format (cic_20.csv)
# =============================================================================
print("\n" + "=" * 60)
print("Generating PIKACHU format (cic_20.csv)...")

# Re-read result with labels
result_pik = pd.read_csv('cic_2017_1.csv', header=0, index_col=0)
result_pik = result_pik.replace(result_pik.iloc[97]['Flow Bytes/s'], 0)
result_pik = result_pik.replace(np.nan, 0)
result_pik.drop(columns=['Flow ID', ' Label', ' Timestamp'], inplace=True)

ti = 20  # 20 minute snapshots for RQ2
result_pik['snapshot'] = result_pik['time_int'] // (60 * ti)

cic = result_pik[['time_int', ' Source IP', ' Destination IP', 'label', 'snapshot']]
cic.columns = ['timestamp', 'src_computer', 'dst_computer', 'label', 'snapshot']
cic.to_csv('cic_20.csv')

# Compute trainwin info
trainwin = result_pik.iloc[529917]['snapshot'] + 1 if len(result_pik) > 529917 else 'N/A'
print(f"  cic_20.csv done. Trainwin for PIKACHU: {trainwin}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("Processing Complete!")
print("=" * 60)
print("Generated files:")
print("  1. cic_2017_1.csv         -> Anomal-E input")
print("  2. cic_argus.csv          -> ARGUS intermediate")
print("  3. argus_flow/            -> ARGUS flow format")
print("  4. euler/                 -> EULER & VGRNN format")
print("  5. cic_20.csv             -> PIKACHU format (20min snapshots)")
print()

# Count output files
argus_files = len([f for f in os.listdir('./argus_flow/') if f.endswith('.txt')])
euler_files = len([f for f in os.listdir('./euler/') if f.endswith('.txt')])
print(f"  argus_flow/ : {argus_files} time-slice files + nmap.pkl")
print(f"  euler/      : {euler_files} time-slice files + nmap.pkl")
