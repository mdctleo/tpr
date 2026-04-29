# PIKACHU on CIC-IDS 2017 — Detailed Technical Analysis

> **Model**: PIKACHU (Paudel et al., 2021)
> **Dataset**: CIC-IDS 2017 (GeneratedLabelledFlows)
> **Task**: Edge-level anomaly detection in temporal interaction graphs
> **Framework**: TensorFlow 2.x + gensim 3.8 (CPU-only)

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Data Processing Pipeline (CIC-IDS 2017 → `cic_20.csv`)](#2-data-processing-pipeline)
3. [Stage 1 — Data Ingestion & Graph Construction](#3-stage-1--data-ingestion--graph-construction)
4. [Stage 2 — Short-Term Embedding via CTDNE](#4-stage-2--short-term-embedding-via-ctdne)
5. [Stage 3 — Long-Term Embedding via GRU Autoencoder](#5-stage-3--long-term-embedding-via-gru-autoencoder)
6. [Stage 4 — Anomaly Detection via Edge Probability](#6-stage-4--anomaly-detection-via-edge-probability)
7. [Hyperparameters & Configuration](#7-hyperparameters--configuration)
8. [Train / Test Split](#8-train--test-split)
9. [Output Files & Metrics](#9-output-files--metrics)
10. [How PIKACHU Differs from EULER and ARGUS](#10-how-pikachu-differs-from-euler-and-argus)

---

## 1. High-Level Architecture

PIKACHU is a **four-stage pipeline** for detecting anomalous edges (network flows) in
a sequence of temporal graphs:

```
┌──────────────────────┐
│  Raw CIC-IDS 2017    │  2,830,742 flows across 5 days
│  (8 CSV files)       │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  cic_20.csv          │  Temporal edge list with 20-min snapshot IDs
│  (columns: timestamp,│  127 unique snapshots (0–312), 19,129 unique IPs
│   src, dst, label,   │
│   snapshot)           │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Stage 1: Graph       │  One NetworkX MultiGraph per snapshot
│ Construction         │  Each flow = one edge (parallel edges allowed)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Stage 2: CTDNE       │  Time-respecting random walks → Word2Vec
│ (Short-Term Emb.)    │  Output: (T, N, d) = (127, 19129, 100) per-snapshot
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Stage 3: GRU-AE      │  Autoencoder over node embedding sequences
│ (Long-Term Emb.)     │  Input: (N, T, d) → Reconstruct → (N, T, d)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Stage 4: Edge Prob.  │  Neighbor aggregation → softmax classifier
│ Anomaly Detection    │  Score each test edge → AUC, AP, TPR, FPR
└──────────────────────┘
```

---

## 2. Data Processing Pipeline

**Source**: `GIDSREP/1-DATA_PROCESSING/cic_2017/process_cic_2017.py`

The raw CIC-IDS 2017 dataset consists of 8 CSV files (Monday through Friday,
some days split into AM/PM). The processing script:

### Step 1 — Load & Parse Timestamps

Each day-file is loaded with `latin-1` encoding. The ` Timestamp` column is parsed
into Unix epoch integers (`time_int`). Rows before 8 AM get a +43200 s offset to
handle overnight wrapping:

```python
# From process_cic_2017.py (Step 1 — Monday example)
mon = pd.read_csv('Monday-WorkingHours.pcap_ISCX.csv', encoding='latin-1', low_memory=False)
time2 = []
for i in mon[' Timestamp'].values:
    tmp = datetime.strptime(i, '%d/%m/%Y %H:%M:%S')
    if tmp.hour < 8:
        time2.append(int(tmp.timestamp()) + 43200)
    else:
        time2.append(int(tmp.timestamp()))
mon['time_int'] = time2
mon1 = mon.sort_values(by='time_int')
```

### Step 2 — Merge All Days & Create Labels

All 8 DataFrames are concatenated, sorted by `time_int`, and rebased to start at 0.
A binary `label` column is derived:

```python
result = pd.concat([mon1, tue1, wed1, Thu_11, Thu_21, Fri_11, Fri_21, Fri_31])
result = result.sort_values(by='time_int').reset_index(drop=True)
result['time_int'] = result['time_int'] - result['time_int'].iloc[0]

label = [1 if i != 'BENIGN' else 0 for i in result[' Label'].values]
result['label'] = label
```

**Result**: 2,830,742 rows, 557,646 attacks, 2,273,096 benign.

### Step 3 — Generate PIKACHU Format (`cic_20.csv`)

The key operation is **snapshot assignment** using 20-minute windows:

```python
# From process_cic_2017.py (OUTPUT 4: PIKACHU format)
ti = 20  # 20 minute snapshots for RQ2
result_pik['snapshot'] = result_pik['time_int'] // (60 * ti)

cic = result_pik[['time_int', ' Source IP', ' Destination IP', 'label', 'snapshot']]
cic.columns = ['timestamp', 'src_computer', 'dst_computer', 'label', 'snapshot']
cic.to_csv('cic_20.csv')
```

**Output file** (`cic_20.csv`):

| Column | Type | Example |
|--------|------|---------|
| `timestamp` | int | 0, 374762 (seconds from start) |
| `src_computer` | str | `192.168.10.5` |
| `dst_computer` | str | `8.254.250.126` |
| `label` | int | 0 (benign) or 1 (attack) |
| `snapshot` | int | 0–312 (20-min window ID) |

**Key statistics**:
- **Total rows (edges)**: 2,830,742
- **Snapshot ID range**: 0–312 (from `time_int // 1200`)
- **Unique non-empty snapshots**: **127 out of 313 possible IDs**
- **Unique IPs**: 19,129 (combined src + dst)
- **Average edges per snapshot**: 22,289 (median 17,351)
- **Min edges**: 186 (snapshot 71) — **Max edges**: 195,807 (snapshot 149)

### Why 127 Snapshots, Not 313?

The snapshot ID is computed as `time_int // (60 × 20)`, which can yield any
integer from 0 to 312. But CIC-IDS 2017 only contains traffic during **working
hours** (~8 AM to 5 PM) on five weekdays. Each day spans roughly 9 hours =
27 twenty-minute windows. Between days (overnight + weekends), there are large
gaps with **zero network flows**, so those snapshot IDs are never assigned to
any row.

Out of the 313 possible IDs, **186 are empty** (no flows at all), leaving
exactly **127 populated snapshots**.

### Day-by-Day Snapshot Breakdown

| Day | Snapshot IDs | # Snapshots | # Edges | # Attacks | Notes |
|-----|-------------|-------------|---------|-----------|-------|
| **Monday** | 0–24 | 25 | 529,918 | 0 | All benign (training data) |
| **Tuesday** | 71–96 | 26 | 445,909 | 13,835 | FTP/SSH brute force |
| **Wednesday** | 143–168 | 26 | 692,703 | 252,672 | DoS slowloris/hulk/goldeneye, Heartbleed |
| **Thursday** | 216–240 | 25 | 458,967 | 2,216 | Web attacks, infiltration |
| **Friday** | 288–312 | 25 | 703,245 | 288,923 | Botnet, PortScan, DDoS |
| **Total** | — | **127** | **2,830,742** | **557,646** | |

The **gaps** between days are:
- Monday ends at snapshot 24, Tuesday starts at 71 → **gap of 46 empty IDs** (overnight Mon→Tue)
- Tuesday ends at 96, Wednesday starts at 143 → **gap of 46 empty IDs** (overnight Tue→Wed)
- Wednesday ends at 168, Thursday starts at 216 → **gap of 47 empty IDs** (overnight Wed→Thu)
- Thursday ends at 240, Friday starts at 288 → **gap of 47 empty IDs** (overnight Thu→Fri)

### Edge Distribution Per Snapshot (All 127)

Most snapshots have 10K–35K edges. Two massive outliers exist:
- **Snapshot 149** (Wed PM): 195,807 edges — DoS Hulk/GoldenEye flood
- **Snapshot 309** (Fri PM): 177,307 edges — DDoS LOIT attack
- **Snapshot 305** (Fri PM): 155,445 edges — PortScan

The smallest is **snapshot 71** (186 edges) — the first few minutes of Tuesday
before traffic ramps up.

---

## 3. Stage 1 — Data Ingestion & Graph Construction

**Files**: `main.py` lines 67–83, `utils.py` (classes `DataUtils` and `GraphUtils`)

### 3.1 Data Loading (`DataUtils.get_data`)

```python
# utils.py — DataUtils.get_data()
class DataUtils:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def get_data(self):
        data_df = pd.read_csv(self.data_folder, header=0)
        node_df = data_df[['src_computer', 'dst_computer']]
        node_df = node_df.drop_duplicates()
        node_map = self.get_node_map(node_df)
        return data_df, node_map
```

This reads the entire `cic_20.csv` and builds a **node map**: a dictionary mapping
each unique IP address to an integer ID (0-indexed). With 19,129 unique IPs, the
node map has 19,129 entries.

```python
# utils.py — DataUtils.get_node_map()
def get_node_map(self, data_df):
    node_map = {}
    node_id = 0
    for index, row in tqdm(data_df.iterrows()):
        scomp = row.src_computer
        dcomp = row.dst_computer
        if scomp not in node_map:
            node_map[scomp] = node_id
            node_id += 1
        if dcomp not in node_map:
            node_map[dcomp] = node_id
            node_id += 1
    return node_map
```

### 3.2 Graph Construction (`GraphUtils.create_graph`)

For each unique snapshot value, a **NetworkX MultiGraph** is created. Every flow
becomes a separate edge — **parallel edges between the same (src, dst) pair are
preserved** (this is a key distinguishing feature vs. EULER/ARGUS):

```python
# utils.py — GraphUtils.create_graph()
def create_graph(self, snapshot_df):
    G = nx.MultiGraph()
    anom_node = []
    for index, row in snapshot_df.iterrows():
        scomp = row.src_computer
        dcomp = row.dst_computer
        time = index       # row index = ordering within CSV
        gid = row.snapshot
        is_anomaly = (row.label == 1)

        if is_anomaly:
            if scomp not in anom_node: anom_node.append(scomp)
            if dcomp not in anom_node: anom_node.append(dcomp)

        G.add_node(self.node_map[scomp], anom=(scomp in anom_node))
        G.add_node(self.node_map[dcomp], anom=(dcomp in anom_node))
        G.add_edge(
            self.node_map[scomp], self.node_map[dcomp],
            time=time, anom=is_anomaly, snapshot=gid, weight=1
        )
    return G
```

**Important details**:
- **Graph type**: `nx.MultiGraph` — allows multiple edges between the same node pair
- **Edge attributes**: `time` (row index for ordering), `anom` (bool), `snapshot` (int), `weight=1`
- **Node attributes**: `anom` (True if any incident edge is anomalous)
- No edge features like duration/bytes/packets — only connectivity + timestamps

**Critical subtlety — the `time` attribute**:

The `time` assigned to each edge is **`index`** from `snapshot_df.iterrows()`,
which is the **DataFrame row index** from the original `cic_20.csv`, NOT the
`timestamp` column (seconds since start). For example, in snapshot 74:
- Row indices range from 581,279 to 615,007
- The `timestamp` column values range from 88,802 to 89,942 (seconds)
- The edge `time` attribute is set to 581,279, 581,280, ... 615,007

This row index preserves the **sequential ordering** of flows as they appeared
in the original dataset, which is what CTDNE uses for its time-respecting walks.

### 3.3 Orchestration in `main.py`

```python
# main.py lines 67–83
dp = DataUtils(data_folder=args.input)     # args.input = "dataset/cic_20.csv"
data_df, node_map = dp.get_data()

g_util = GraphUtils(node_map)
graphs = []
for t in tqdm(data_df.snapshot.unique()):
    graphs.append(g_util.create_graph(data_df[data_df['snapshot'] == t]))

# Persist to disk
with open('weights/graphs' + data_file, 'wb') as f:
    pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### 3.4 What Are the 127 MultiGraphs?

The loop iterates over `data_df.snapshot.unique()` — the 127 non-empty snapshot
IDs. Each call to `create_graph()` filters the CSV to that snapshot's rows and
builds one `nx.MultiGraph`. The result is a **Python list of 127 MultiGraph
objects**, indexed 0 through 126:

```
graphs[0]  → MultiGraph for snapshot ID 0   (Monday, first 20 min)    — 11,734 edges
graphs[1]  → MultiGraph for snapshot ID 1   (Monday, 20–40 min)       — 24,970 edges
...
graphs[24] → MultiGraph for snapshot ID 24  (Monday, last window)     — 4,327 edges
graphs[25] → MultiGraph for snapshot ID 71  (Tuesday, first window)   — 186 edges
graphs[26] → MultiGraph for snapshot ID 72  (Tuesday, second window)  — 19,459 edges
...
graphs[126]→ MultiGraph for snapshot ID 312 (Friday, last window)     — 4,069 edges
```

**The list index (0–126) does NOT equal the snapshot ID (0–312).** The list is
ordered by the order `snapshot.unique()` returns them (which follows the CSV
ordering, i.e., chronological). So `graphs[25]` is the 26th graph in the list
but corresponds to snapshot ID 71 (first 20 min of Tuesday).

Each MultiGraph contains:
- **Nodes**: only IPs that appear in that snapshot (not all 19,129)
- **Edges**: every individual flow as a separate edge (parallel edges between
  same IP pair allowed)
- **Total edges across all 127 graphs**: 2,830,742 (same as CSV rows — no
  flow is dropped or merged)

Serialized as `weights/graphs_cic_20_sample.pickle`.

### 3.5 End-to-End Timestamp Lifecycle for a Single Edge

Let's trace one concrete flow through the entire pipeline:

**Raw CSV row** (CIC-IDS 2017, Tuesday-WorkingHours):
```
Flow ID: 192.168.10.9-151.101.44.207-...
Timestamp: 04/07/2017 09:33  (human-readable)
Source IP: 192.168.10.9
Destination IP: 151.101.44.207
Label: BENIGN
```

**Step 1 — `process_cic_2017.py`** converts the timestamp:
```
time_int = unix_epoch("04/07/2017 09:33") - base_time = 88802  (seconds from Monday 8am)
snapshot = 88802 // 1200 = 74
```
This row is written to `cic_20.csv` as:
```
581279, 88802, 192.168.10.9, 151.101.44.207, 0, 74
       ^timestamp                              ^label ^snapshot
```
(The 581279 is its position in the merged+sorted CSV = the DataFrame index.)

**Step 2 — `utils.py create_graph()`** builds a MultiGraph for snapshot 74:
```python
G.add_edge(
    node_map["192.168.10.9"],      # e.g., integer 42
    node_map["151.101.44.207"],     # e.g., integer 1837
    time=581279,                     # ← DataFrame row index, NOT 88802
    anom=False,
    snapshot=74,
    weight=1
)
```

**Step 3 — `ctdne.py _precompute_probabilities()`** collects edge times per neighbor:
```python
# For node 42, neighbor 1837:
neighbor2times[1837] = [581279, 581295, ...]   # all edge times between them
```

**Step 4 — `parallel.py` time-respecting walk** uses the `time` attribute:
```python
# Walk is at node 42, last_time = 581200
# Consider edges to neighbor 1837:
walk_options += [(1837, prob, 581279)]   # 581279 > 581200 ✓ valid
walk_options += [(1837, prob, 581295)]   # 581295 > 581200 ✓ valid
# If last_time were 581290:
walk_options += [(1837, prob, 581279)]   # 581279 < 581290 ✗ FILTERED OUT
walk_options += [(1837, prob, 581295)]   # 581295 > 581290 ✓ valid
```

**Step 5 — Word2Vec**: The walk sequence `["42", "1837", "503", ...]` is fed
to skip-gram. The time values are no longer used — they served their purpose
in constraining walk order.

**Step 6 — GRU Autoencoder**: Takes the per-snapshot Word2Vec embedding of
node 42 at snapshot 74 (a 100-dim vector) as one time step in node 42's
sequence across all 127 snapshots.

**Step 7 — Anomaly Detection**: If snapshot 74 is in the test set, the
long-term embedding of node 42 is used to compute $P(1837 | 42)$ via
the softmax classifier. A low probability means this edge is anomalous.

**Summary**: The original timestamp (88802 seconds) determines which snapshot
a flow belongs to (snapshot 74). The DataFrame row index (581279) becomes the
edge's `time` attribute, used solely by CTDNE to enforce temporal ordering
during random walks. After the walks, the time information is consumed — it
is not explicitly present in the final embeddings.

---

## 4. Stage 2 — Short-Term Embedding via CTDNE

**Files**: `pikachu.py` (function `short_term_embedding`, method `learn_embedding`),
`CTDNE/ctdne.py`, `CTDNE/parallel.py`

### 4.1 What is CTDNE?

**Continuous-Time Dynamic Network Embedding** (Nguyen et al., 2018) extends
Node2Vec by making random walks **time-respecting**: each step in the walk must
move forward in time. This ensures the walk captures the temporal causality of
interactions.

### 4.2 Per-Snapshot Embedding

For **each** of the 127 snapshot graphs, PIKACHU runs CTDNE independently:

```python
# pikachu.py — short_term_embedding() (top-level function)
def short_term_embedding(args, node_list, idx, G):
    CTDNE_model = CTDNE(G,
        dimensions=args.dimensions,  # 100
        workers=4,
        walk_length=args.walklen,    # 500
        num_walks=args.numwalk,      # 1
        quiet=True)
    ctdne_model = CTDNE_model.fit(window=10, min_count=1, batch_words=4)

    # Map Word2Vec vectors back to fixed node ordering
    node_embs = np.zeros((len(node_list), args.dimensions), dtype=np.float32)
    for i in range(len(ctdne_model.wv.vocab)):
        if ctdne_model.wv.index2word[i] in node_list:
            node_vec = ctdne_model.wv[ctdne_model.wv.index2word[i]]
            node_embs[node_list.index(ctdne_model.wv.index2word[i])] = np.array(node_vec)
    return node_embs
```

### 4.3 Time-Respecting Random Walks

The core of CTDNE is in `CTDNE/parallel.py`. Unlike standard Node2Vec, each walk
step only considers edges with timestamps **strictly greater than** the last step's
timestamp:

```python
# CTDNE/parallel.py — inside parallel_generate_walks()
while len(walk) < walk_length:
    if len(walk) == 1:
        probabilities = d_graph[walk[-1]][first_travel_key]
    else:
        probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]

    walk_options = []
    for neighbor, p in zip(d_graph[walk[-1]].get(neighbors_key, []), probabilities):
        times = d_graph[walk[-1]][neighbors_time_key][neighbor]
        # *** KEY: Only edges with time > last_time are valid ***
        walk_options += [(neighbor, p, time) for time in times if time > last_time]

    if len(walk_options) == 0:
        break  # Dead end — no future edges

    # Linear time-decay weighting (default)
    if use_linear:
        time_probabilities = np.array(
            np.argsort(np.argsort(list(map(lambda x: x[2], walk_options)))[::-1]) + 1,
            dtype=np.float)
        final_probabilities = time_probabilities * np.array(
            list(map(lambda x: x[1], walk_options)))
        final_probabilities /= sum(final_probabilities)

    walk_to_idx = np.random.choice(range(len(walk_options)), size=1, p=final_probabilities)[0]
    walk_to = walk_options[walk_to_idx]
    last_time = walk_to[2]
    walk.append(walk_to[0])
```

**Key mechanism**: The `time > last_time` filter ensures temporal causality.
The walk can only traverse edges that occurred **after** the previous edge,
mimicking real information flow in the network.

### 4.4 Transition Probability Precomputation

Before walks begin, `CTDNE._precompute_probabilities()` builds a dictionary
storing neighbors, transition probabilities (using Node2Vec's p/q parameters),
and **per-neighbor timestamp lists**:

```python
# CTDNE/ctdne.py — _precompute_probabilities()
# For each (current_node, neighbor) pair, collect all edge timestamps:
neighbor2times = {}
for neighbor in d_neighbors:
    neighbor2times[neighbor] = []
    if 'time' in self.graph[current_node][neighbor]:
        neighbor2times[neighbor].append(self.graph[current_node][neighbor]['time'])
    else:
        # MultiGraph: iterate over parallel edges
        for att in list(self.graph[current_node][neighbor].values()):
            neighbor2times[neighbor].append(att['time'])
d_graph[current_node][self.NEIGHBORS_TIME_KEY] = neighbor2times
```

This is where **MultiGraph parallel edges matter**: between the same (src, dst)
there may be dozens of flows at different times, and **each becomes a valid walk
option at its respective timestamp**.

### 4.5 Word2Vec (Skip-Gram)

The collected walks are fed to gensim's Word2Vec:

```python
# CTDNE/ctdne.py — fit()
return gensim.models.Word2Vec(self.walks, **skip_gram_params)
# skip_gram_params: size=100, window=10, min_count=1, workers=4
```

### 4.6 Parallel Execution & Output Shape

All snapshots are processed in parallel (1 CPU core in practice):

```python
# pikachu.py — learn_embedding()
data_tuple = [(self.args, self.node_list, idx, G) for idx, G in enumerate(self.graphs)]
with Pool(total_cpu) as pool:
    short_term_embs = pool.starmap(short_term_embedding, data_tuple)

short_term_embs = np.array(short_term_embs)
# Shape: (T, N, d) = (127, 19129, 100)
```

**Output**: A 3D tensor of shape **(127 snapshots × 19,129 nodes × 100 dimensions)**.
Nodes not present in a snapshot get zero vectors (from the `np.zeros` init).

Saved as `weights/short_term_cic_20_sample_d100.pickle`.

---

## 5. Stage 3 — Long-Term Embedding via GRU Autoencoder

**File**: `pikachu.py` (methods `autoencoder_model`, `long_term_embedding`)

### 5.1 Purpose

Short-term embeddings capture each node's structural role **within** a single
snapshot. The GRU autoencoder learns **how each node's embedding evolves over
time**, compressing the temporal trajectory into a fixed representation.

### 5.2 Transpose: (T, N, d) → (N, T, d)

```python
# pikachu.py — learn_embedding()
short_term_embs = np.transpose(short_term_embs, (1, 0, 2))
# Now shape: (19129, 127, 100) — one sequence per node
```

Each node now has a **time-series of 127 embedding vectors** (one per snapshot).

### 5.3 GRU Autoencoder Architecture

```python
# pikachu.py — autoencoder_model()
def autoencoder_model(self, time_step, dim):
    # time_step = 127 (snapshots), dim = 100 (embedding dimensions)
    input = Input(shape=(time_step, dim))          # (127, 100)
    mask = Masking(mask_value=0.)(input)            # Mask zero-padded absent nodes

    # ===== ENCODER =====
    el1 = GRU(64, return_sequences=True)(mask)     # → (127, 64)
    do_enc = Dropout(0.3)(el1)
    encoded = GRU(128, return_sequences=False)(do_enc)  # → (128,) bottleneck

    # ===== DECODER =====
    rp = RepeatVector(time_step)(encoded)           # → (127, 128)
    dl1 = GRU(128, return_sequences=True)(rp)       # → (127, 128)
    do_de = Dropout(0.3)(dl1)
    dl3 = GRU(64, return_sequences=True)(do_de)     # → (127, 64)
    decoded = TimeDistributed(Dense(dim))(dl3)       # → (127, 100) reconstruct

    return Model(inputs=input, outputs=decoded), \
           Model(inputs=input, outputs=encoded)
```

**Architecture summary**:

```
Input (127, 100)
  → Masking(0.0)
  → GRU(64, return_sequences=True)    [encoder layer 1]
  → Dropout(0.3)
  → GRU(128, return_sequences=False)  [bottleneck = 128-dim]
  → RepeatVector(127)
  → GRU(128, return_sequences=True)   [decoder layer 1]
  → Dropout(0.3)
  → GRU(64, return_sequences=True)    [decoder layer 2]
  → TimeDistributed(Dense(100))       [reconstruct original dim]
Output (127, 100)
```

### 5.4 Training

```python
# pikachu.py — long_term_embedding()
self.model.compile(optimizer='adam', loss='mse')
history = self.model.fit(
    short_term_embs,      # X = node embedding sequences
    short_term_embs,      # Y = same (autoencoder reconstructs input)
    epochs=self.args.epoch,   # 50
    verbose=1,
    validation_split=0.1      # 10% of nodes used for validation
)
graph_embedding = self.model.predict(short_term_embs)
return graph_embedding    # Shape: (19129, 127, 100)
```

**Key points**:
- Loss = MSE (mean squared error between input and reconstruction)
- Optimizer = Adam
- 90% of nodes for training, 10% for validation
- 50 epochs
- The **reconstructed** embeddings (not the bottleneck) are used downstream

Saved as `weights/long_term_cic_20_sample_d100.pickle`.

---

## 6. Stage 4 — Anomaly Detection via Edge Probability

**File**: `anomaly_detection.py` (class `AnomalyDetection`)

### 6.1 Core Idea

For each edge (u, v) in the test graphs, PIKACHU asks:
> "Given node u's neighborhood, how probable is it that v is a neighbor?"

Low probability → the edge is **anomalous**.

### 6.2 Neighbor Aggregation

```python
# anomaly_detection.py — aggregate_neighbors()
def aggregate_neighbors(node_emb, node_list, u, n_u):
    CNu = [node_emb[node_list.index(str(n))] for n in n_u]
    Cu = node_emb[node_list.index(str(u))]
    H = 1 / (1 + len(CNu)) * np.add(Cu, sum(Cn for Cn in CNu))
    return np.array(H)
```

This computes a **mean-pooled neighborhood representation**:

$$H_u = \frac{1}{1 + |N_u|} \left( \mathbf{e}_u + \sum_{n \in N_u} \mathbf{e}_n \right)$$

where $\mathbf{e}_u$ is node $u$'s long-term embedding and $N_u$ is a support set
of $s$ neighbors (sampled with replacement).

### 6.3 Training the Edge Probability Classifier

The classifier is a simple **softmax linear layer**: $P = \text{softmax}(H \cdot W^T)$
where $W \in \mathbb{R}^{|V| \times d}$.

```python
# anomaly_detection.py — get_train_edges()
def get_train_edges(self, train_graphs, s=10):
    data_x, data_y = [], []
    for G in tqdm(train_graphs):
        for u in G.nodes():
            N = [n for n in G.neighbors(u)]
            for v in N:
                if len(N) > 1:
                    n_minus_v = [n for n in N if n != v]
                    support_set = random.choices(n_minus_v, k=s)
                else:
                    support_set = random.choices(N, k=s)
                H = self.aggregate_neighbors_object(u, support_set)
                y = np.zeros(len(self.node_list))  # One-hot: 19,129 classes
                y[v] = 1
                data_x.append(H)
                data_y.append(y)
        self.idx += 1
    return np.array(data_x), np.array(data_y)
```

**Training procedure**:
- For each node `u` in each training graph, for each neighbor `v`:
  - Sample `s=15` neighbors of `u` (excluding `v`)
  - Compute aggregated representation `H_u`
  - Target: one-hot vector with `y[v] = 1`
- Optimization: manual gradient descent on cross-entropy loss

```python
# anomaly_detection.py — gradient_descent()
def gradient_descent(self, w, X, Y, iterations, learning_rate):
    for i in range(iterations):
        grads, cost = self.propagate(w, X, Y)
        w = w - learning_rate * grads["dw"]
    return {"w": w}, costs

# propagate() computes: dw = (1/m) * (softmax(X @ W^T) - Y)^T @ X
```

Parameters: `iterations=10`, `learning_rate=0.001`, weight matrix $W$ is
$19129 \times 100$.

### 6.4 Test-Time Edge Scoring

For each edge (u, v) in test graphs (snapshots after `trainwin`):

```python
# anomaly_detection.py — calculate_edge_probability()
def calculate_edge_probability(w, node_emb, node_list, G):
    edge_scores = []
    for u, v, data in G.edges(data=True):
        Nu = [n for n in G.neighbors(u)]
        Hu = aggregate_neighbors(node_emb, node_list, u, Nu)
        Pv = predict_prob(w, Hu.reshape(1, -1))   # softmax(Hu @ W^T)

        Nv = [n for n in G.neighbors(v)]
        Hv = aggregate_neighbors(node_emb, node_list, v, Nv)
        Pu = predict_prob(w, Hv.reshape(1, -1))   # softmax(Hv @ W^T)

        # Anomaly score = average of (1 - P(v|u)) and (1 - P(u|v))
        score = ((1 - Pv[0, v]) + (1 - Pu[0, u])) / 2
        edge_scores.append([u, v, score, data['snapshot'], data['time'], data['anom']])
    return edge_scores
```

**Score interpretation**:
- $\text{score} = \frac{(1 - P(v|u)) + (1 - P(u|v))}{2}$
- High score → edge is unlikely given neighborhood → **anomalous**
- The scoring is **symmetric**: both directions are checked

### 6.5 Threshold & Metrics

```python
# anomaly_detection.py — calculate_performance_metrics()
fpr, tpr, thresholds = metrics.roc_curve(true_label, scores, pos_label=1)

# Optimal cutoff using weighted FPR/TPR balance
fw = 0.5   # FPR weight
tw = 1 - fw  # TPR weight
fn = np.abs(tw * tpr - fw * (1 - fpr))
best = np.argmin(fn, 0)
```

Reports: AUC, AP (Average Precision), TPR, FPR, Precision, Recall, Confusion Matrix.

---

## 7. Hyperparameters & Configuration

### Command Line (from `cic.sh`)

```bash
python main.py \
    --dataset cic_20 \
    --input dataset/cic_20.csv \
    --trainwin 25 \
    --dimensions 100 \
    --alpha 0.001 \
    --support 15
```

### Full Parameter Table

| Parameter | Flag | Value | Description |
|-----------|------|-------|-------------|
| **Dataset name** | `--dataset` | `cic_20` | Used in weight/result file names |
| **Input file** | `--input` | `dataset/cic_20.csv` | Path to edge list CSV |
| **Train window** | `--trainwin` | 25 | Number of snapshots for training (0–24) |
| **Embedding dim** | `--dimensions` | 100 | CTDNE / Word2Vec vector size |
| **Walk length** | `--walklen` | 500 | Max nodes per CTDNE random walk |
| **Num walks** | `--numwalk` | 1 | Random walks per node per snapshot |
| **GRU epochs** | `--epoch` | 50 | Autoencoder training epochs |
| **GD iterations** | `--iter` | 10 | Gradient descent steps for edge prob. |
| **Learning rate** | `--alpha` | 0.001 | Edge probability classifier LR |
| **Support set** | `--support` | 15 | Neighbors sampled for aggregation |
| **Word2Vec window** | (hardcoded) | 10 | Skip-gram context window |
| **GRU hidden** | (hardcoded) | 64 / 128 | Encoder: GRU(64)→GRU(128), Decoder: GRU(128)→GRU(64) |
| **Dropout** | (hardcoded) | 0.3 | Dropout rate in autoencoder |
| **AE validation** | (hardcoded) | 0.1 | 10% nodes held out for AE validation |
| **p, q** | (default) | 1, 1 | Node2Vec return/in-out parameters |

---

## 8. Train / Test Split

PIKACHU uses a **temporal split** based on the `--trainwin` parameter:

```
Snapshot IDs:   0   1   2   ...  24  |  25  26  ...  312
                ├── TRAIN ──────────┤  ├── TEST ────────┤
                (first 25 snapshots)    (remaining snapshots)
```

- **Training snapshots (0–24)**: Used to train the edge probability classifier
  (gradient descent on $W$). These are all **benign** traffic.
- **Test snapshots (25–312)**: Edge probability scores are computed for every
  edge in these graphs. The first attacks appear at **snapshot 73**.

**Note**: The GRU autoencoder sees **all** 127 snapshots during its unsupervised
training (it reconstructs node embedding sequences). Only the edge probability
classifier uses the train/test split.

---

## 9. Output Files & Metrics

### Weight Files (in `weights/`)

| File | Shape | Description |
|------|-------|-------------|
| `node_map_cic_20_sample.pickle` | dict (19,129 entries) | IP → integer ID mapping |
| `graphs_cic_20_sample.pickle` | list of 127 MultiGraphs | Per-snapshot graph objects |
| `short_term_cic_20_sample_d100.pickle` | (127, 19129, 100) | CTDNE embeddings per snapshot |
| `long_term_cic_20_sample_d100.pickle` | (19129, 127, 100) | GRU-AE reconstructed embeddings |
| `param_cic_20_sample_d100_0.001_15.pickle` | W: (19129, 100) | Trained edge probability weights |

### Results (in `results/`)

| File | Description |
|------|-------------|
| `results.txt` | AUC, AP, FPR, TPR, P, tn/fp/fn/tp, cutoff, timing |
| `cic_20_d100all_users.csv` | All test edges with scores, predictions, src/dst IPs |

---

## 10. How PIKACHU Differs from EULER and ARGUS

### 10.1 Graph Representation

| Aspect | PIKACHU | EULER | ARGUS |
|--------|---------|-------|-------|
| **Graph type** | `nx.MultiGraph` | PyG `Data` (simple graph) | PyG `Data` (simple graph) |
| **Parallel edges** | ✅ Each flow is a separate edge | ❌ Edges deduplicated, counted | ❌ Edges deduplicated |
| **Edge features** | None (only `time`, `weight=1`) | Edge count only (frequency) | O_cic: none; L_cic_flow: mean/std of dur, bytes, pkts (6 features) |
| **Node features** | None (identity via Word2Vec) | One-hot `eye(N)` | One-hot `eye(N)` |
| **Snapshot granularity** | 20-minute windows | `FILE_DELTA = 100000` seconds (~27.8 hours) | `delta` seconds (10–15 configurable) |
| **Framework** | TensorFlow + NetworkX + gensim | PyTorch + PyG + torch.distributed | PyTorch + PyG |

### 10.2 How Each Model Uses Timestamps

**PIKACHU** — Timestamps are **first-class citizens**:
- Each flow keeps its original row-index as `time` on the edge
- CTDNE random walks enforce `time > last_time` at every step
- The walk literally follows the **causal order** of network events

**EULER** — Timestamps define **time slices only**:
- Flows are bucketed into `FILE_DELTA=100000` second windows
- Within a window, all edges are simultaneous (no intra-window ordering)
- The GCN processes a **sequence of graph snapshots** (3–4 snapshots total)

**ARGUS** — Timestamps define **delta-second windows**:
- Flows within each `delta`-second window are aggregated per (src, dst) pair
- Edge features (mean/std of duration, bytes, packets) summarize the window
- Temporal modeling via `NNConv` + GRU across window sequence

### 10.3 Flow Aggregation

**PIKACHU** — **No aggregation**:
- Every individual flow is a separate edge in the MultiGraph
- If IP-A sends 100 flows to IP-B in one snapshot, there are 100 parallel edges
- The CTDNE walk can traverse each flow independently based on its timestamp

**EULER** — **Frequency aggregation**:
- All flows between (src, dst) in a time slice become **one edge**
- The edge weight = number of flows (frequency count)
- Flow-level features (duration, bytes, packets) are **discarded**

**ARGUS** — **Statistical aggregation** (L_cic_flow variant):
- All flows between (src, dst) in a delta-window are merged into one edge
- 6 edge features computed: `mean_dur, std_dur, mean_pkts, std_pkts, mean_bytes, std_bytes`
- The O_cic variant discards features entirely (like EULER)

### 10.4 Embedding Method

| Aspect | PIKACHU | EULER | ARGUS |
|--------|---------|-------|-------|
| **Node embedding** | CTDNE (temporal random walk + Word2Vec) | GCN encoder (1-layer + BatchNorm + PReLU) | NNConv (edge-conditioned message passing) |
| **Temporal modeling** | GRU autoencoder over embedding sequences | GRU over graph-level readouts | GRU hidden state across time windows |
| **Embedding dim** | 100 (configurable) | 16–32 (configurable `-z`) | 16–32 (configurable `-z`) |
| **Unsupervised objective** | MSE reconstruction (AE) + Skip-gram (W2V) | Link prediction: score positive vs negative edges | Link prediction: AUC loss (libauc) |

### 10.5 Anomaly Scoring

**PIKACHU** — **Edge probability** via softmax classifier:
- $\text{score}(u,v) = \frac{(1 - P(v|u)) + (1 - P(u|v))}{2}$
- Trained on benign-only snapshots (0–24)
- Scores every individual edge in test snapshots

**EULER** — **Embedding distance** change detection:
- Computes edge embeddings at each time step
- Detects anomalies by comparing test-time edge scores against training distribution
- Uses GCN + GRU to predict "expected" embeddings

**ARGUS** — **AUC-based classification**:
- Trains an inner-product decoder to predict edge existence
- Uses AUCM loss (libauc) to optimize AUC directly
- `fpweight` controls false-positive vs false-negative trade-off
- Threshold tuned on validation set; binary classification of edges

### 10.6 Scale & Computational Profile

| Metric | PIKACHU | EULER | ARGUS |
|--------|---------|-------|-------|
| **Nodes** | 19,129 | 64,650 (euler/ format) | 19,129 (argus_flow/) or 64,650 (euler/) |
| **Snapshots** | 127 | 3–4 (large time slices) | Varies by delta (10–15 sec windows → many) |
| **Graph framework** | NetworkX (in-memory) | PyG (sparse tensors) | PyG (sparse tensors) |
| **Parallel execution** | `multiprocessing.Pool` | `torch.distributed.rpc` (6 workers) | Single process |
| **GPU** | ❌ CPU-only (`CUDA_VISIBLE_DEVICES="-1"`) | ❌ CPU-only | ❌ CPU-only (can use GPU) |
| **Bottleneck** | CTDNE walks (500-length × 19K nodes × 127 snapshots) | RPC worker communication | NNConv message passing per time step |

### 10.7 Summary Comparison Table

| Feature | PIKACHU | EULER | ARGUS |
|---------|---------|-------|-------|
| **Approach** | Walk-based + AE + Edge Prob | GCN + GRU link prediction | NNConv + GRU + AUC loss |
| **Keeps individual flows?** | ✅ Yes (MultiGraph) | ❌ No (counts only) | ❌ No (aggregated stats) |
| **Uses flow features?** | ❌ No | ❌ No | ✅ Yes (L_cic_flow) / ❌ No (O_cic) |
| **Time-aware walks?** | ✅ Yes (CTDNE) | ❌ No walks | ❌ No walks |
| **Temporal model** | GRU autoencoder | GRU over GCN outputs | GRU over NNConv outputs |
| **Anomaly signal** | Edge probability (softmax) | Edge embedding distance | Edge prediction (AUC) |
| **Train data** | First K benign snapshots | Temporal link prediction | Temporal link prediction |
| **Snapshot size** | 20 minutes | ~27.8 hours | 10–15 seconds |

---

## Appendix: File Structure

```
PIKACHU/
├── main.py                     # Entry point: parse → graph → embed → detect
├── pikachu.py                  # CTDNE orchestration + GRU autoencoder
├── anomaly_detection.py        # Edge probability training & scoring
├── utils.py                    # DataUtils (CSV loading) + GraphUtils (MultiGraph)
├── cic.sh                      # One-liner run command with CIC-IDS parameters
├── CTDNE/
│   ├── __init__.py
│   ├── ctdne.py                # CTDNE: time-aware transition probabilities + walks
│   ├── parallel.py             # Time-respecting random walk generation
│   ├── node2vec.py             # Base Node2Vec (not time-aware, unused by CTDNE)
│   ├── edges.py                # Edge embedding utilities (Hadamard, L1, L2, Avg)
│   └── setup.cfg
├── dataset/
│   └── cic_20.csv → symlink    # Points to 1-DATA_PROCESSING/cic_2017/cic_20.csv
├── weights/                    # Serialized embeddings & parameters
│   ├── node_map_cic_20_sample.pickle
│   ├── graphs_cic_20_sample.pickle
│   ├── short_term_cic_20_sample_d100.pickle
│   ├── long_term_cic_20_sample_d100.pickle
│   └── param_cic_20_sample_d100_0.001_15.pickle
└── results/
    └── results.txt             # AUC, AP, FPR, TPR, confusion matrix
```

---

## 11. Proposed Enhancement: KDE Timestamp-Diff Edge Features (Option A)

### 11.1 Goal

Add a **20-dimensional KDE vector per unique (src, dst) edge** that encodes the
distribution of inter-arrival times (timestamp differences) for that pair. This
gives the anomaly detector a "timing fingerprint" for each communication channel.

### 11.2 Scope & Constraints

| Item | Detail |
|------|--------|
| **Feature** | KDE of sorted timestamp differences (inter-arrival times) |
| **Granularity** | One vector per unique `(src_ip, dst_ip)` pair — same across all snapshots |
| **Dimension** | 20 (configurable `rkhs_dim`) |
| **Training data only** | KDEs constructed from **train snapshots only** (list indices 0–24, i.e., snapshot IDs 0–24 = Monday). Test timestamps must NOT leak into the KDE. |
| **Eligibility threshold** | Only edges with **>10 distinct timestamps** in the training set get a real KDE vector |
| **Fallback** | Edges with ≤10 timestamps get a **zeros vector**: `np.zeros(20)` (neutral "no information" signal) |
| **Integration point** | Concatenated to the neighbor-aggregation vector in `anomaly_detection.py` (Option A from Section discussion) |

### 11.3 Why Timestamp Differences (Not Raw Timestamps)?

Raw timestamps for a single (src, dst) pair within Monday might be:
`[100, 105, 106, 200, 201, 202, 500]`. The **absolute values** (100, 500) are
specific to when Monday's capture started — they don't generalize to Tuesday.

The **differences** `[5, 1, 94, 1, 1, 298]` capture the **inter-arrival pattern**:
"this pair bursts (1 s apart), then goes quiet (94–298 s)". This pattern is
transferable across days and is exactly what the `kde_diff` mode in
`kde_computation.py` computes.

### 11.4 KDE Construction Method (from `kde_computation.py`)

The construction follows the `fit_batched_gpu_dpgmm` pipeline but adapted for
CPU-only execution (no GPU available in PIKACHU's environment):

```
For each unique (src, dst) pair in train snapshots:
  1. Collect all edge timestamps from cic_20.csv where snapshot ∈ train set
  2. Sort timestamps → compute consecutive absolute differences
  3. If len(diffs) ≤ 10: assign ones(20) and skip
  4. Scale diffs with MaxAbsScaler (divide by max absolute value → [-1, 1])
  5. Fit scipy.stats.gaussian_kde on scaled diffs (bandwidth='scott')
  6. Evaluate KDE on a uniform grid of 20 points spanning [min, max] of scaled data
  7. The 20-dim density vector = KDE feature for this (src, dst) pair
```

The `KDEVectorComputer.kde_to_rkhs_vector()` fallback in `kde_computation.py`
uses a richer feature vector (quadrature + moments + Fourier + quantiles). For
simplicity and consistency we use the **grid-evaluated density** approach from
the GPU path, but on CPU via `scipy.stats.gaussian_kde`.

### 11.5 Detailed Plan

#### New file: `compute_kde_features.py`

A standalone preprocessing script (run once before `main.py`). It:

1. **Reads** `cic_20.csv`
2. **Filters** to training snapshots only (snapshot IDs corresponding to
   list indices 0 through `trainwin-1`). Since `data_df.snapshot.unique()`
   returns `[0, 1, 2, ..., 24, 71, 72, ...]`, the first 25 entries are
   snapshot IDs 0–24 (Monday). So the filter is simply `snapshot <= 24`.
3. **Groups** by `(src_computer, dst_computer)` and collects all `timestamp`
   values for each pair
4. **Computes diffs**: `np.abs(np.diff(np.sort(timestamps)))`
5. **Filters** pairs with `len(diffs) > 10`
6. **Fits KDE** per pair: `gaussian_kde(scaled_diffs, bw_method='scott')`
7. **Evaluates** on 20-point uniform grid → 20-dim vector
8. **L2-normalizes** each vector: `vec / (np.linalg.norm(vec) + 1e-8)`
9. **Assigns fallback** `np.zeros(20)` for pairs with ≤10 diffs
10. **Saves** `weights/kde_edge_features.pickle`:
   ```python
   {
       (src_ip, dst_ip): np.array([...], dtype=np.float32),  # shape (20,)
       ...
   }
   ```

#### Changes to existing files

**`anomaly_detection.py`** — Two functions change:

1. **`AnomalyDetection.__init__()`** — accept and store the KDE dict:
   ```python
   def __init__(self, args, node_list, node_map, node_embeddings, idx, kde_features=None):
       ...
       self.kde_features = kde_features or {}
       self.kde_dim = 20
   ```

2. **`aggregate_neighbors()` and `aggregate_neighbors_object()`** — concatenate
   KDE vector when computing the neighbor aggregation:
   ```python
   def aggregate_neighbors_with_kde(self, u, v, n_u):
       # Original node embedding aggregation (100-dim)
       H = self.aggregate_neighbors_object(u, n_u)

       # Look up KDE vector for this (u, v) edge
       src_ip = self.get_ip(u)
       dst_ip = self.get_ip(v)
       kde_vec = self.kde_features.get((src_ip, dst_ip), np.zeros(self.kde_dim))

       # Concatenate: 100 + 20 = 120-dim
       return np.concatenate([H, kde_vec])
   ```

3. **`get_train_edges()`** — produce 120-dim training vectors:
   ```python
   # Instead of:
   H = self.aggregate_neighbors_object(u, support_set)
   # Use:
   H = self.aggregate_neighbors_with_kde(u, v, support_set)
   ```
   The target `y` stays the same (one-hot over nodes, shape `(19129,)`).

4. **`calculate_edge_probability()`** (top-level function) — also needs the KDE
   dict to produce 120-dim inputs. The weight matrix $W$ becomes $(19129 \times 120)$.

5. **`initialize_parameters()`** — change `k` from `args.dimensions` to
   `args.dimensions + kde_dim`:
   ```python
   w = self.initialize_parameters(self.args.dimensions + self.kde_dim, len(self.node_list))
   # W shape: (19129, 120)
   ```

**`main.py`** — load the KDE features and pass them through:
```python
# After loading long_term_embs:
kde_path = 'weights/kde_edge_features.pickle'
if os.path.exists(kde_path):
    with open(kde_path, 'rb') as f:
        kde_features = pickle.load(f)
else:
    kde_features = {}

ad_long_term = AnomalyDetection(args, node_list, node_map, long_term_embs, idx=0,
                                 kde_features=kde_features)
```

**No changes needed to**: `pikachu.py`, `CTDNE/`, `utils.py`, or the GRU autoencoder.
The KDE feature is injected **only at Stage 4** (anomaly detection), not into the
embedding pipeline.

### 11.6 Data Flow Diagram (Modified Pipeline)

```
                    ┌──────────────────────────────────────────────┐
                    │  compute_kde_features.py  (run once, offline)│
                    │                                              │
                    │  cic_20.csv (train snapshots 0–24 only)      │
                    │      ↓                                       │
                    │  Group by (src, dst) → collect timestamps     │
                    │      ↓                                       │
                    │  sorted diffs → MaxAbsScale → KDE fit         │
                    │      ↓                                       │
                    │  Evaluate on 20-point grid → 20-dim vector    │
                    │      ↓                                       │
                    │  kde_edge_features.pickle                     │
                    │  { (ip_a, ip_b): array(20,), ... }           │
                    └──────────────────┬───────────────────────────┘
                                       │
    ┌──────────────────────────────────┐│┌─────────────────────────────────┐
    │ Stages 1–3 (UNCHANGED)          │││ Stage 4: Anomaly Detection      │
    │                                  │││ (MODIFIED)                      │
    │ Graph Construction → CTDNE       │││                                 │
    │    → GRU-AE → long_term_embs    ├┘│ For each test edge (u, v):      │
    │                                  │ │   node_agg = mean_pool(embs)    │
    │ Output: node_embs (N, T, 100)   ├─→   kde_vec = lookup(src, dst)    │
    │                                  │ │   H = concat(node_agg, kde_vec) │
    └──────────────────────────────────┘ │       = 120-dim                 │
                                         │   P(v|u) = softmax(H @ W^T)    │
                                         │   W shape: (N, 120)            │
                                         └─────────────────────────────────┘
```

### 11.7 Final Decisions (User-Specified)

The following design decisions were finalized based on user requirements:

1. **KDE method**: Use **DPGMM pure density-on-grid** approach (NOT moments+Fourier+quantiles).
   Fit `scipy.stats.gaussian_kde` with Scott bandwidth, then evaluate on 20 uniform grid
   points to produce the feature vector.

2. **Edge directionality**: Treat `(A → B)` and `(B → A)` as **separate directed edges**,
   each with their own KDE vector. This preserves the src/dst distinction from CIC-IDS flows.

3. **Fallback value**: Use `np.zeros(20)` for edges with ≤10 timestamps (NOT ones).
   Zeros serve as a neutral "no timing information available" signal.

4. **Test-time new edges**: Edges appearing for the first time in test snapshots receive
   the `zeros(20)` fallback. This is expected behavior — unknown edges have no timing history.

5. **Normalization**: **L2-normalize** KDE vectors before concatenation with 100-dim node
   embeddings. This prevents scale imbalance between KDE density values (~0–5 range) and
   Word2Vec embeddings.

6. **Configuration flag**: Add `--kde` argument in `main.py` to conditionally enable KDE
   features. When `--kde` is False (default), run base PIKACHU unchanged. When True, load
   pre-computed KDE features and use 120-dim vectors in anomaly detection.

### 11.8 Training Data Coverage Analysis

Analysis of which nodes and edge pairs appear in the Monday training snapshots (0–24):

| Metric | Count | Percentage |
|--------|-------|------------|
| Total unique nodes (all data) | 19,129 | 100% |
| Nodes in training (snapshots 0–24) | 9,709 | **50.76%** |
| Nodes NOT in training | 9,420 | 49.24% |
| | | |
| Total unique (src,dst) pairs (all data) | 113,769 | 100% |
| Pairs in training (snapshots 0–24) | 39,404 | **34.64%** |
| Pairs NOT in training | 74,365 | 65.36% |

**Implications**: Approximately 65% of edge pairs in test data will receive the
`zeros(20)` fallback because they never appear in Monday's training snapshots. This
is by design — these are "unknown" communication patterns from the detector's perspective.
### 11.9 Usage Instructions

#### Step 1: Compute KDE Features (run once)
```bash
cd /scratch/asawan15/GIDSREP/2-DETECTION_ASSESSMENT/RQ2/cic_2017/PIKACHU

# Compute KDE features from training snapshots (Monday, snapshots 0-24)
python compute_kde_features.py \
    --input dataset/cic/cic_20.csv \
    --trainwin 25 \
    --kde_dim 20 \
    --output weights/kde_edge_features.pickle
```

#### Step 2a: Run Base PIKACHU (no KDE)
```bash
# Default mode - unchanged behavior
python main.py \
    -ip dataset/cic/ \
    -d cic_20 \
    -k 100 \
    -w 25 \
    -e 50 \
    -i 10 \
    -r 0.001 \
    -s 15
```

#### Step 2b: Run KDE-Enhanced PIKACHU
```bash
# Enable KDE features with --kde flag
python main.py \
    -ip dataset/cic/ \
    -d cic_20 \
    -k 100 \
    -w 25 \
    -e 50 \
    -i 10 \
    -r 0.001 \
    -s 15 \
    --kde \
    --kde_file weights/kde_edge_features.pickle \
    --kde_dim 20
```

#### Running Both in One Job
```bash
#!/bin/bash
# run_both_pikachu.sh

# Ensure KDE features are computed
if [ ! -f weights/kde_edge_features.pickle ]; then
    echo "Computing KDE features..."
    python compute_kde_features.py --input dataset/cic/cic_20.csv
fi

# Run base PIKACHU
echo "Running base PIKACHU..."
python main.py -ip dataset/cic/ -d cic_20 -k 100 -w 25 -e 50 -i 10 -r 0.001 -s 15

# Run KDE-enhanced PIKACHU (reuses embeddings, only re-trains Stage 4)
echo "Running KDE-enhanced PIKACHU..."
python main.py -ip dataset/cic/ -d cic_20 -k 100 -w 25 -e 50 -i 10 -r 0.001 -s 15 --kde
```

### 11.10 Implementation Files

| File | Purpose |
|------|---------|
| `compute_kde_features.py` | NEW: Offline preprocessing script to compute KDE vectors |
| `main.py` | MODIFIED: Added `--kde`, `--kde_file`, `--kde_dim` arguments |
| `anomaly_detection.py` | MODIFIED: Added `aggregate_neighbors_with_kde()`, KDE-aware training/inference |

### 11.11 Expected Behavior

| Mode | Feature Dimension | W Matrix Shape | Result File |
|------|-------------------|----------------|-------------|
| Base PIKACHU | 100 | (19129, 100) | `cic_20_d100all_users.csv` |
| KDE-Enhanced | 120 | (19129, 120) | `cic_20_d100_kdeall_users.csv` |

Both modes share the same Stages 1–3 (graph construction, CTDNE, GRU-AE). Only
Stage 4 (anomaly detection) differs — the KDE mode uses 120-dim input vectors
instead of 100-dim.