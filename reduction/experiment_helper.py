from create_graph_memory import fit_edges_memory, filter_edges_by_count, create_and_merge_graph, average_of_edges
from tqdm import tqdm
from enhanced_edge import EnhancedEdge
from datetime import datetime

import glob
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import os

def reduce_helper(csv_path, target_nodes, args):
    
    technique_name = ""
    if args.use_dpgmm:
        technique_name = "dpgmm"
    elif args.use_kmeans:
        technique_name = "kmeans"
    elif args.use_dbstream:
        technique_name = "dbstream"
    else:
        technique_name = "kde"


    technique = {
        technique_name: {
            "neg_average_log_likelihood": [],
            "integrated_square_error": [],
            "mean_squared_error": [],
            "kl_divergence": [],
            "timestamp_count": [],
            "num_clusters": []
        }
    }

    min_edge_count = 200
    

    files = glob.glob(csv_path)

    merged_edges, total_edge_count = create_and_merge_graph(files, target_nodes, args)
    print("Total edge count: ", total_edge_count, flush=True)
    print("Num of merged edges: ", len(merged_edges), flush=True)        
    
    merged_edges = filter_edges_by_count(merged_edges, min_edge_count)
    print("Num of merged edges: ", len(merged_edges), flush=True)
    fit_edges_memory(merged_edges, args)

    for edge in merged_edges.values():
        neg_log_likelihood = edge.compute_average_log_likelihood()
        integrated_square_error = edge.compute_integrated_square_error()

        if neg_log_likelihood is None or integrated_square_error is None:
            continue

        technique[technique_name]["neg_average_log_likelihood"].append(edge.compute_average_log_likelihood())
        technique[technique_name]["integrated_square_error"].append(edge.compute_integrated_square_error())
        technique[technique_name]["kl_divergence"].append(edge.compute_kl_divergence())
        technique[technique_name]["num_clusters"].append(edge.num_clusters)
        technique[technique_name]["timestamp_count"].append(len(edge.timestamps))
        edge.visualize_distribution(name=f"{technique_name}_{csv_path.split('/')[-1].split('.')[0]}")

    average_likelihood = average_of_edges(technique[technique_name]["neg_average_log_likelihood"])
    average_integrated_square_error = average_of_edges(technique[technique_name]["integrated_square_error"])
    average_num_clusters = average_of_edges(technique[technique_name]["num_clusters"])
    average_kl_divergence = average_of_edges(technique[technique_name]["kl_divergence"])

    print(f"=== Results for {csv_path} using {technique_name} ===", flush=True)
    print(f"{technique_name} average likelihood: ", average_likelihood, flush=True)
    print(f"{technique_name} integrated square error: ", average_integrated_square_error, flush=True)
    print(f"{technique_name} average kl divergence: ", average_kl_divergence, flush=True)
    print(f"{technique_name} average num clusters: ", average_num_clusters, flush=True)

def reduce(dataset_name, max_components=100, grid_points=400, total_count=2000, method="scan_gmm"):
        
    dataset_path = f"./datasets/{dataset_name}/{method}.pkl"
    if os.path.exists(dataset_path):
        print(f"Loading preprocessed dataset from {dataset_path}", flush=True)
        with open(dataset_path, "rb") as f:
            enhanced_edges = pickle.load(f)
    
        return enhanced_edges
    
    dataset_path = f"./datasets/{dataset_name}/*.csv"

    args = {}
    args.sys_call = True
    args.method = method
    args.zero_center = False
    args.detection = False
    args.visualize = False
    args.K = max_components
    args.grid_points = grid_points
    args.total_count = total_count

    merged_edges, scalers = prep_data(dataset_path, None, args)
    enhanced_edges = fit_data(merged_edges, args)

    save_path = f"./datasets/{dataset_name}/{method}.pkl"
    edges_dict = enhanced_edges_to_pickle_dict(save_path, enhanced_edges, scalers)
    
    return edges_dict

    
def prep_data(csv_path, target_nodes, args):
    files = []
    if type(csv_path) is list:
        for path in csv_path:
            files.extend(glob.glob(path))
    else:
        files = glob.glob(csv_path)

    merged_edges, total_edge_count = create_and_merge_graph(files, target_nodes, args)

    print("Total edge count: ", total_edge_count, flush=True) 
    print("Number of merged edges: ", len(merged_edges), flush=True)

    merged_edges = filter_edges_by_count(merged_edges, min_count=200)

    edge_counts = np.array([len(edge) for edge in merged_edges.values()])
    if edge_counts.size > 0:
        avg_edge_count_pre_filter = float(np.mean(edge_counts))
        median_edge_count_pre_filter = float(np.median(edge_counts))
        min_edge_count_pre_filter = int(np.min(edge_counts))
        max_edge_count_pre_filter = int(np.max(edge_counts))
    else:
        avg_edge_count_pre_filter = median_edge_count_pre_filter = 0.0
        min_edge_count_pre_filter = max_edge_count_pre_filter = 0

    print(f"Edge count statistics before filtering: avg={avg_edge_count_pre_filter}, median={median_edge_count_pre_filter}, min={min_edge_count_pre_filter}, max={max_edge_count_pre_filter}", flush=True)
    

    # merged_edges = filter_edges_by_count(merged_edges, min_count=avg_edge_count_pre_filter)

    scalers = {}
    for key, timestamps in merged_edges.items():
        # sort the data
        merged_edges[key] = sorted(timestamps)
        merged_edges[key] = np.array(timestamps).reshape(-1, 1)
        
        scaler = MinMaxScaler()
        merged_edges[key] = scaler.fit_transform(merged_edges[key])
        scalers[key] = scaler
        # print("max after scaler: ", np.max(merged_edges[key]), flush=True)
        # print("min after scaler: ", np.min(merged_edges[key]), flush=True)

    print("Number of merged edges after filtering: ", len(merged_edges), flush=True)
    
    return merged_edges, scalers

def fit_data(merged_edges, args):
    """
    Fit the merged edges to the EnhancedEdge model.

    Parameters:
    merged_edges (dict): Dictionary with keys as tuples of node IDs and syscall, and values as lists of timestamps.
    args: Arguments containing configuration for the fitting process.

    Returns:
    dict: A dictionary of EnhancedEdge objects fitted with the provided timestamps.
    """
    enhanced_edges = {}
    
    for (node1_id, node2_id, syscall), timestamps in tqdm(merged_edges.items(), total=len(merged_edges), desc="Fitting enhanced edges..."):
        enhanced_edge = EnhancedEdge(u=node1_id, v=node2_id, syscall=syscall, args=args)
        enhanced_edge.fit(timestamps=timestamps, args=args, data_weights=None)
        enhanced_edges[(node1_id, node2_id, syscall)] = enhanced_edge

    print("Number of enhanced edges after fitting: ", len(enhanced_edges), flush=True)

    return enhanced_edges

def enhanced_edges_to_pickle_dict(name, enhanced_edges, scalers):
    edge_dict = {}
    for key, enhanced_edge in enhanced_edges.items():
        edge_dict[key] = {
            "kde": enhanced_edge.kde,
            "scaler": scalers[key],
            "means": enhanced_edge.means if enhanced_edge.means is not None else [],
            "covariances": enhanced_edge.covariances if enhanced_edge.covariances is not None else [],
            "weights": enhanced_edge.weights if enhanced_edge.weights is not None else [],
            "num_clusters": enhanced_edge.num_clusters if enhanced_edge.num_clusters is not None else 0
        }
    
    with open(f"{name}", "wb") as f:
        pickle.dump(edge_dict, f)

    return edge_dict
        
def split_data(merged_edges, k=20000):
    """
    Split the values of each dictionary item into chunks of approximately K data points,
    and return a list of dictionaries, each containing chunks for the same keys.

    Parameters:
    merged_edges (dict): Dictionary with keys and array values to split.
    k (int): Number of data points per chunk.

    Returns:
    list of dict: A list of dictionaries, each containing chunks of data for the same keys.
    """
    # Initialize a list to hold the split dictionaries
    split_dicts = []

    # Iterate over each key-value pair in the dictionary
    for key, values in merged_edges.items():
        # Convert values to a NumPy array if not already
        values = np.array(values)

        if len(values) < 2000:
            k = len(values)  # If fewer than 2000 points, use all points as a single chunk
        else:
            # k = max(1, len(values) // 4)  # Ensure k is at least 1
            k = 40000
        
        # Split the values into chunks of size K
        chunks = [values[i:i + k] for i in range(0, len(values), k)]
        
        # Ensure there are enough dictionaries in the split_dicts list
        while len(split_dicts) < len(chunks):
            split_dicts.append({})

        # Assign each chunk to the corresponding dictionary in split_dicts
        for i, chunk in enumerate(chunks):
            split_dicts[i][key] = chunk

    # Print split statistics for debugging
    for i, split in enumerate(split_dicts):
        total_points = sum(len(values) for values in split.values())
        print(f"Split {i + 1}: {len(split)} keys, {total_points} total points", flush=True)

    return split_dicts

def split_data_cic_ids(merged_edges, scaler, args):
    time_ranges = [
        (1499054400, 1499140799),  # Monday
        (1499140800, 1499227199),  # Tuesday
        (1499227200, 1499313599),  # Wednesday
        (1499400000, 1499486399)   # Friday
    ]

    scaled_time_ranges = [
        (scaler.transform([[start]])[0][0], scaler.transform([[end]])[0][0])
        for start, end in time_ranges
    ]

    # Initialize a list to hold the split dictionaries
    split_dicts = [{} for _ in range(len(scaled_time_ranges))]

    # Iterate over each key-value pair in the dictionary
    for key, timestamps in merged_edges.items():
        for i, (start, end) in enumerate(scaled_time_ranges):
            # Filter timestamps that fall within the current time range
            split_dicts[i][key] = np.array(timestamps)[(start <= np.array(timestamps)) & (np.array(timestamps) <= end)]
            split_dicts[i][key] = split_dicts[i][key].reshape(-1, 1)  # Reshape to 2D array 
    return split_dicts


def process_splits(splits, args):
    enhanced_edges = {}

    # Step 1: Fit each split individually
    split_enhanced_edges = []
    for index, split in tqdm(enumerate(splits), total=len(splits), desc="Fitting individual splits..."):
        split_edges = {}
        for (node1_id, node2_id, syscall), timestamps in tqdm(split.items(), total=len(split), desc=f"Fitting split {index}..."):
            args.K = 20  # Adjust K for the initial fit
            enhanced_edge = EnhancedEdge(u=node1_id, v=node2_id, syscall=syscall, args=args)
            enhanced_edge.fit(timestamps=timestamps, args=args, data_weights=None)
            split_edges[(node1_id, node2_id, syscall)] = enhanced_edge
        split_enhanced_edges.append(split_edges)

        evaluate_enhancement(f"split_{index}", split_edges, split, args)

 # Step 2: Extract GH3 pseudo-points from all splits
    combined_gh3_points = {}
    combined_gh3_weights = {}
    for split_edges in split_enhanced_edges:
        for key, edge in split_edges.items():
            if key not in combined_gh3_points:
                combined_gh3_points[key] = []
                combined_gh3_weights[key] = []
            gh3_points, gh3_weights = edge.gh3_pseudo_points()
            combined_gh3_points[key].append(gh3_points)
            combined_gh3_weights[key].append(gh3_weights)


    # Step 3: Combine GH3 pseudo-points and fit a unified distribution
    for key in combined_gh3_points.keys():
        # Combine GH3 pseudo-points and weights for this edge
        all_points = np.concatenate(combined_gh3_points[key], axis=0)
        all_weights = np.concatenate(combined_gh3_weights[key], axis=0)

        # Fit a single unified distribution for this edge
        args.K = 100
        enhanced_edge = EnhancedEdge(u=key[0], v=key[1], syscall=key[2], args=args)
        enhanced_edge.fit(timestamps=all_points, args=args, data_weights=all_weights)
        enhanced_edges[key] = enhanced_edge


    return enhanced_edges

def evaluate_enhancement(csv_path, enhanced_edges, merged_edges, args):

    csv_name = csv_path.split("/")[-1].split(".")[0]
        
    technique_name = ""
    if args.use_dpgmm:
        technique_name = "dpgmm"
    elif args.use_kmeans:
        technique_name = "kmeans"
    elif args.use_dbstream:
        technique_name = "dbstream"
    else:
        technique_name = "kde"

    technique = {
        technique_name: {
            "neg_average_log_likelihood": [],
            "integrated_square_error": [],
            "mean_squared_error": [],
            "kl_divergence": [],
            "wasserstein_distance": [],
            "timestamp_count": [],
            "num_clusters": []
        }
    }

    visualization_count = 0
    
    for key, edge in enhanced_edges.items():
        all_timestamps = merged_edges[key]
        # print("all timstamps shape: ", all_timestamps.shape, flush=True)
    
        neg_log_likelihood = edge.compute_average_log_likelihood(all_timestamps)
        kl_divergence = edge.compute_kl_divergence(all_timestamps)
        wasserstein_dist = edge.compute_wasserstein_distance(all_timestamps)
        integrated_square_error = edge.compute_integrated_square_error(all_timestamps)

        if neg_log_likelihood is None or kl_divergence is None:
            continue

        technique[technique_name]["neg_average_log_likelihood"].append(neg_log_likelihood)
        technique[technique_name]["kl_divergence"].append(kl_divergence)
        technique[technique_name]["wasserstein_distance"].append(wasserstein_dist)
        technique[technique_name]["num_clusters"].append(edge.num_clusters)
        technique[technique_name]["integrated_square_error"].append(integrated_square_error)
        if args.visualize and visualization_count < 10:
            visualization_count += 1
            print("Got inside visualization for edge: ", flush=True)
            edge.visualize_distribution(timestamps=all_timestamps, name=f"./kde_visualizations/{csv_name}_{technique_name}_{edge.u}_{edge.v}.png")

        # technique[technique_name]["timestamp_count"].append(len(edge.timestamps))

    average_likelihood = average_of_edges(technique[technique_name]["neg_average_log_likelihood"])
    average_num_clusters = average_of_edges(technique[technique_name]["num_clusters"])
    average_kl_divergence = average_of_edges(technique[technique_name]["kl_divergence"])
    average_wasserstein_distance = average_of_edges(technique[technique_name]["wasserstein_distance"])
    average_integrated_square_error = average_of_edges(technique[technique_name]["integrated_square_error"])

    print(f"=== Results for {csv_name} using {technique_name} ===", flush=True)
    print(f"{technique_name} average likelihood: ", average_likelihood, flush=True)
    print(f"{technique_name} average kl divergence: ", average_kl_divergence, flush=True)
    print(f"{technique_name} average num clusters: ", average_num_clusters, flush=True)
    print(f"{technique_name} average wasserstein distance: ", average_wasserstein_distance, flush=True)
    print(f"{technique_name} integrated square error: ", average_integrated_square_error, flush=True)


def set_adversarial_edge_timestamps(target_start_time, target_end_time, merged_edges, scalers):
    attacker_victim_pair = ("172.16.0.1_same", "192.168.10.50_same", "same")
    target_start_timestamp = int(datetime.strptime(target_start_time, '%Y-%m-%d %H:%M:%S').timestamp())
    target_end_timestamp = int(datetime.strptime(target_end_time, '%Y-%m-%d %H:%M:%S').timestamp())
    # Ensure scalers are in the same order as merged_edges
    for (key, timestamps), scaler in zip(merged_edges.items(), scalers):
        # Inverse transform the scaled timestamps to get the original timestamps
        original_timestamps = scaler.inverse_transform(np.array(timestamps).reshape(-1, 1)).flatten()

        if attacker_victim_pair == key:
            print(f"Shifting timestamps for edge: {attacker_victim_pair}", flush=True)

            # Replace the earliest and latest timestamps with the target values
            shifted_timestamps = np.copy(original_timestamps)
            shifted_timestamps[np.argmin(original_timestamps)] = target_start_timestamp
            shifted_timestamps[np.argmax(original_timestamps)] = target_end_timestamp

            # Transform the updated timestamps back to the scaled range
            scaled_shifted_timestamps = scaler.transform(shifted_timestamps.reshape(-1, 1))

            # Update the timestamps in merged_edges
            merged_edges[attacker_victim_pair] = scaled_shifted_timestamps

            # Convert the shifted timestamps to human-readable format for verification
            start_time_human = datetime.fromtimestamp(min(shifted_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
            end_time_human = datetime.fromtimestamp(max(shifted_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Shifted Edge: {attacker_victim_pair}, Start Time: {start_time_human}, End Time: {end_time_human}", flush=True)
        else:
            # For all other edges, just print the human-readable timestamps
            start_time_human = datetime.fromtimestamp(min(original_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
            end_time_human = datetime.fromtimestamp(max(original_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Edge: {key}, Start Time: {start_time_human}, End Time: {end_time_human}", flush=True)
