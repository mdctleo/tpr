

import pandas as pd
from tqdm import tqdm
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import numpy as np



def read_csvs(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def create_and_merge_graph(file_paths, target_nodes, args):
    merged_edges = defaultdict(list)
    total_edge_count = 0
    unique_nodes = set()

    print(f"Reading and merging {len(file_paths)} file(s)...", flush=True)

    for file_index, file_path in enumerate(tqdm(file_paths, desc="files"), start=1):
        print(f"Processing file {file_index}/{len(file_paths)}: {file_path}", flush=True)

        chunk_iter = pd.read_csv(file_path, chunksize=100000)
        for chunk_index, df in enumerate(tqdm(chunk_iter, desc=f"chunks[{file_index}]", leave=False), start=1):
            df["node1_id"] = df["subjectname"]
            if "subject_type" in df.columns:
                df["node1_id"] = df["node1_id"] + "_" + df["subject_type"]

            df["node2_id"] = df["objectname"]
            if "object_type" in df.columns:
                df["node2_id"] = df["node2_id"] + "_" + df["object_type"]

            total_edge_count += len(df)
            unique_nodes.update(df["node1_id"].unique())
            unique_nodes.update(df["node2_id"].unique())

            if args.detection:
                df = df[df["objectname"].isin(target_nodes)].copy()
                df["syscall"] = "same"
                df["subjectname"] = "same"
                df["subject_type"] = "same"
                df["object_type"] = "same"
            elif args.static_detection:
                df = df[df["objectname"].isin(target_nodes)].copy()
                df["subject_type"] = "same"
                df["object_type"] = "same"
                df["syscall"] = "same"
            else:
                df["timestamp"] = df["timestamp"] / 1e9

            df = df[["node1_id", "node2_id", "syscall", "timestamp"]]

            if target_nodes is not None and not (args.detection or args.static_detection):
                df = df[df["node2_id"].str.rsplit("_", n=1).str[0].isin(target_nodes)]

            grouped = (
                df.groupby(["node1_id", "node2_id", "syscall"], sort=False)["timestamp"]
                .agg(list)
            )

            for key, timestamps in grouped.items():
                merged_edges[key].extend(timestamps)

            print(
                f"  merged chunk {chunk_index}: rows={len(df)}, chunk_groups={len(grouped)}, total_merged_edges={len(merged_edges)}",
                flush=True,
            )

    print("Total number of unique nodes: ", len(unique_nodes), flush=True)
    print("Total number of edges: ", total_edge_count, flush=True)
    return merged_edges, total_edge_count


def merge_edges_memory(nodes, edges, args):
    print("Merging edges...", flush=True)
    merged_edges = {}

    for edge in tqdm(edges, total=len(edges), desc="merging edges..."):
        key = (edge.u, edge.v)
        if args.sys_call:
            key = (edge.u, edge.v, edge.syscall)

        if key not in merged_edges:
            merged_edges[key] = edge
        else:
            merged_edges[key].update_params(edge.timestamp, edge.syscall, args)

    print("finished merging edges!", flush=True)
    print("number of merged edges: ", len(merged_edges), flush=True)

    return merged_edges

def fit_edges_memory(merged_edges, scaler, args):
    print("Fitting edges...", flush=True)

    for (u, v, syscall), edge in tqdm(merged_edges.items(), total=len(merged_edges), desc="fitting edges..."):
        # if args.online_detection:
        #     edge.fit(args, old_merged_edges)
        # else:
        edge.fit(scaler, args)

    print("finished fitting edges!", flush=True)

# Given the target nodes, fetch the target nodes and
# its immediate neighbors
def subgraph(merged_edges, target_nodes, num_hops=0):
    print("Number of edges: ", len(merged_edges), flush=True)

    # print(merged_edges.keys(), flush=True)

    # for target_node in target_nodes:
    #     assert target_node in merged_edges, f"Node {target_node} not found in the graph"
    
    subgraph = {}
    for key, edge in merged_edges.items():
        # if target_nodes == [] and len(edge.timestamps) >= 200:
        #     subgraph[key] = edge

        if any(edge.v == target_node for target_node in target_nodes) and len(edge.timestamps) >= 1000:
            subgraph[key] = edge

    print("Number of edges after subgraphing: ", len(subgraph), flush=True)
    
    # subgraph = {}
    # for key, edge in merged_edges.items():
    #     if num_hops == 0:
    #         if edge.u == target_nodes[0] and edge.v == target_nodes[0]:
    #             subgraph[key] = edge
    #     elif num_hops == 1:
    #         if any(edge.u.find(target_node) != -1 or edge.v.find(target_node) != -1 for target_node in target_nodes) and len(edge.timestamps) >= 2:
    #             subgraph[key] = edge

    # print("Number of nodes after subgraphing: ", len(subgraph), flush=True)


    return subgraph


def plot_subgraph(subgraph, name):
    G = nx.MultiDiGraph()

    # Add edges to the graph
    for key, edge in subgraph.items():
        G.add_edge(edge.u, edge.v)

    # Draw the graph
    plt.figure(figsize=(15, 10))
    pos = nx.kamada_kawai_layout(G)  # You can use different layouts like spring_layout, circular_layout, etc.
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=10, font_color='black', edge_color='gray')
    plt.title(name)
    plt.savefig(f"./{name}.png")


def save_to_pickle(path, dictionary):
    with open(path, 'wb') as f:
        pickle.dump(dictionary, f)
    
    print(f"Saved to {path}", flush=True)

def sort_edges_by_timestamp_count(merged_edges, min_count=2):
    sorted_edges = []
    # print(merged_edges.values(), flush=True)
    for edge in merged_edges.values():
        if edge.count >= min_count:
            sorted_edges.append((len(edge.timestamps), edge))
    
    sorted_edges.sort(key=lambda x: x[0])
    return sorted_edges


def filter_edges_by_count(merged_edges, min_count=2):
    filtered_edges = {}
    for key, edge in merged_edges.items():
        if len(edge) >= min_count:
            filtered_edges[key] = edge

    return filtered_edges

# def filter_edges_by_count(merged_edges, min_count=2):
#     # Find the edge with the most timestamps
#     max_edge = None
#     max_timestamps = 0

#     for key, edge in merged_edges.items():
#         if len(edge) > max_timestamps:
#             max_timestamps = len(edge)
#             max_edge = {key: edge}

#     return max_edge


def merge_two_merged_edges(merged_edges1, merged_edges2):
    final_merged_edges = merged_edges1.copy()
    for key, edge in merged_edges2.items():
        if key in final_merged_edges:
            final_merged_edges[key] = np.concatenate((final_merged_edges[key], edge))
        else:
            final_merged_edges[key] = edge

        final_merged_edges[key] = final_merged_edges[key].reshape(-1, 1)  # Ensure the edge is a 2D array
        print(f"Edge {key} has shape: {final_merged_edges[key].shape}", flush=True)

    return final_merged_edges


def separate_two_merged_edges(merged_edges, separate_time_early, separate_time_later):
    earlier_edges = {}
    later_edges = {}

    for key, timestamps in merged_edges.items():
        print("timestamps shape: ", timestamps.shape, flush=True)
        # Use NumPy boolean indexing to separate timestamps and reshape to preserve the second dimension
        earlier_edges[key] = timestamps[timestamps <= separate_time_early].reshape(-1, 1)
        later_edges[key] = timestamps[timestamps > separate_time_later].reshape(-1, 1)

        print("earliest edge: ", earlier_edges[key].shape, flush=True)
        print("latest edge: ", later_edges[key].shape, flush=True)

    return earlier_edges, later_edges


def average_of_edges(list_of_metrics):
    total_metrics = 0
    count = 0
    for likelihood in list_of_metrics:
        total_metrics += likelihood
        count += 1
    
    return total_metrics / count
    
