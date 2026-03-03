import argparse
from utils import ks_test_kdes
import pickle
from detection import create_edge_features, evaluate_autoencoder, create_true_labels, train_baseline
import numpy as np
import pandas as pd
from create_graph_memory import merge_two_merged_edges, separate_two_merged_edges
from datetime import datetime
from online_detection import train_recency_detector, detect_recency_detector
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from experiment_helper import reduce_helper, split_data, prep_data, process_splits, evaluate_enhancement, split_data_cic_ids, fit_data, set_adversarial_edge_timestamps
import random
import torch
import pyro

def set_random_seed(seed):
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)

    pyro.set_rng_seed(seed)  # For Pyro, if used in the project
    
    # Set PyTorch random seed (if using PyTorch)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU-based operations
    
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_random_seed(42)
    
    parser = argparse.ArgumentParser(description="Experiment configs")
    parser.add_argument("--experiment_type", type=str, help="Type of experiment to run")
    parser.add_argument("--distribution", type=str, help="Distribution to use for edge reduction, kde or poisson")
    parser.add_argument("--dataset", type=str, help="dataset to use for graph reduction E5_Trace or E5_Cadet")
    parser.add_argument("--num_files", type=int, help="Number of files to read")
    parser.add_argument("--truncate_object_name", type=bool, default=False, help="Whether to truncate object name or not, truncating creates more overalpping edges")
    parser.add_argument("--truncate_time", default=False, action="store_true", help="Whether to truncate time or not, it creates a smllaer interval")
    parser.add_argument("--sys_call", default=False, action="store_true", help="Whether to use system when reducing graph")
    parser.add_argument("--kernel", type=str, help="Kernel to use for KDE")
    parser.add_argument("--use_dpgmm", default=False, action="store_true", help="Whether to use DPGMM or not")
    parser.add_argument("--minimize_cluster_technique", type=str, help="Technique to use for minimizing clusters")
    parser.add_argument("--minimize_cluster_technique_k", type=int, help="K value to use for minimizing clusters")
    parser.add_argument("--minimize_cluster_variance_threshold", type=float, help="Variance threshold to use for minimizing clusters")
    parser.add_argument("--minimize_cluster_should_merge", default=False, action="store_true", help="Whether to merge clusters or not")
    parser.add_argument("--detection", default=False, action="store_true", help="Whether we are using detection dataset or not")
    parser.add_argument("--online_detection", default=False, action="store_true", help="Whether we are using online detection or not")
    parser.add_argument("--use_weighted_dpgmm", default=False, action="store_true", help="Whether to use weighted DPGMM or not")
    parser.add_argument("--use_window", default=False, action="store_true", help="Whether to use windowed DPGMM or not")
    parser.add_argument("--use_dbstream", default=False, action="store_true", help="Whether to use DBStream or not")
    parser.add_argument("--use_kmeans", default=False, action="store_true", help="Whether to use KMeans or not")
    parser.add_argument("--visualize", default=False, action="store_true", help="Whether to visualize the results or not")
    parser.add_argument("--static_detection", default=False, action="store_true", help="Whether to use static detection or not")
    parser.add_argument("--use_density", default=False, action="store_true", help="Whether to use density or not")

    args = parser.parse_args()


    if args.experiment_type == "reduction_kde_e5_trace":
        args.kernel = "gaussian"
        args.sys_call = True
        args.zero_center = False
        args.detection = False
        args.visualize = True

        merged_edges, _ = prep_data("./datasets/E5_Trace/*.csv", None, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/E5_Trace/*.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_dpgmm_e5_trace":

        args.sys_call = True
        args.use_dpgmm = True
        args.zero_center = False
        args.detection = False
        # args.visualize = True
        args.K = 100

        merged_edges, _ = prep_data("./datasets/E5_Trace/*.csv", None, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/E5_Trace/*.csv", enhanced_edges, merged_edges, args)


    elif args.experiment_type == "reduction_dbstream_e5_trace":
        args.sys_call = True
        args.zero_center = False
        args.detection = False
        args.use_dbstream = True

        # reduce_helper("./datasets/E5_Trace/*.csv", None, args)
        merged_edges, _ = prep_data("./datasets/E5_Trace/*.csv", None, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/E5_Trace/*.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_kmeans_e5_trace":
        args.sys_call = True
        args.zero_center = False
        args.detection = False
        args.use_kmeans = True
        args.visualize = True   

        merged_edges, _ = prep_data("./datasets/E5_Trace/*.csv", None, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/E5_Trace/*.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_kde_e5_cadets":
        args.kernel = "gaussian"
        args.sys_call = True
        args.zero_center = False
        args.detection = False

        merged_edges, _ = prep_data("./datasets/E5_Cadets/*.csv", None, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/E5_Cadets/*.csv", enhanced_edges, merged_edges, args)
        
    elif args.experiment_type == "reduction_dpgmm_e5_cadets":
        args.sys_call = True
        args.use_dpgmm = True
        args.zero_center = False
        args.detection = False
        args.K = 100

        # reduce_helper("./datasets/E5_Cadets/*.csv", None, args)
        merged_edges, _ = prep_data("./datasets/E5_Cadets/*.csv", None, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/E5_Cadets/*.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_dbstream_e5_cadets":
        args.sys_call = True
        args.zero_center = False
        args.detection = False
        args.use_dbstream = True

        # reduce_helper("./datasets/E5_Cadets/*.csv", None, args)
        merged_edges, _ = prep_data("./datasets/E5_Cadets/*.csv", None, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/E5_Cadets/*.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_kmeans_e5_cadets":
        args.sys_call = True
        args.zero_center = False
        args.detection = False
        args.use_kmeans = True

        merged_edges, _ = prep_data("./datasets/E5_Cadets/*.csv", None, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/E5_Cadets/*.csv", enhanced_edges, merged_edges, args)
            
    elif args.experiment_type == "reduction_dpgmm_monday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        # args.detection = True
        args.visualize = True
        args.K = 100

        target_nodes = ["192.168.10.50"]
        # target_nodes = []      

        merged_edges, scalers = prep_data("./datasets/cic-ids-2017/Monday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)

        key = list(enhanced_edges.keys())[0]
        timestamps = merged_edges[key]
        scaler = scalers[0]

        evaluate_enhancement("./datasets/cic-ids-2017/Monday-WorkingHours.csv", enhanced_edges, merged_edges, args)
        enhanced_edges[key].visualize_distribution(name="monday_temporal_pattern", timestamps=timestamps, scaler=scaler)


    elif args.experiment_type == "reduction_kde_monday_cic_ids":
        args.kernel = "gaussian"
        args.sys_call = True
        args.zero_center = False
        args.detection = True
        args.visualize = True

        target_nodes = ["192.168.10.50"]
        reduce_helper("./datasets/cic-ids-2017/Monday-WorkingHours.csv", target_nodes, args)

    elif args.experiment_type == "reduction_kmeans_monday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_kmeans = True
        args.detection = True

        target_nodes = ["192.168.10.50"]
        reduce_helper("./datasets/cic-ids-2017/Monday-WorkingHours.csv", target_nodes, args)

    elif args.experiment_type == "reduction_dbstream_monday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dbstream = True
        args.detection = True

        target_nodes = ["192.168.10.50"]
        merged_edges, _ = prep_data("./datasets/cic-ids-2017/Monday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/cic-ids-2017/Monday-WorkingHours.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_kde_tuesday_cic_ids":
        args.kernel = "gaussian"
        args.sys_call = True
        args.zero_center = False
        args.detection = True

        target_nodes = ["192.168.10.50"]

        merged_edges, _ = prep_data("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_dbstream_tuesday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dbstream = True
        args.detection = True
        args.visualize = True

        target_nodes = ["192.168.10.50"]
        merged_edges, _ = prep_data("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_kmeans_tuesday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_kmeans = True
        args.detection = True

        target_nodes = ["192.168.10.50"]
        merged_edges, _ = prep_data("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_dpgmm_tuesday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = True
        args.visualize = True
        args.K = 100

        target_nodes = ["192.168.10.50"]

        # reduce_helper("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", target_nodes, args)
        merged_edges, scalers = prep_data("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)

        key = list(enhanced_edges.keys())[0]
        timestamps = merged_edges[key]
        scaler = scalers[0]

        evaluate_enhancement("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", enhanced_edges, merged_edges, args)

        enhanced_edges[key].visualize_distribution(name="tuesday_temporal_pattern_update", timestamps=timestamps, scaler=scaler)

        


    elif args.experiment_type == "reduction_dpgmm_wednesday_cic_ids":

        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = True
        args.K = 100

        target_nodes = ["192.168.10.50"]
        merged_edges, _ = prep_data("./datasets/cic-ids-2017/Wednesday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/cic-ids-2017/Wednesday-WorkingHours.csv", enhanced_edges, merged_edges, args)
    
    elif args.experiment_type == "reduction_kde_wednesday_cic_ids":
        args.kernel = "gaussian"
        args.sys_call = True
        args.zero_center = False
        args.detection = True

        target_nodes = ["192.168.10.50"]
        reduce_helper("./datasets/cic-ids-2017/Wednesday-WorkingHours.csv", target_nodes, args)

    elif args.experiment_type == "reduction_kmeans_wednesday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_kmeans = True
        args.detection = True

        target_nodes = ["192.168.10.50"]
        reduce_helper("./datasets/cic-ids-2017/Wednesday-WorkingHours.csv", target_nodes, args)

    elif args.experiment_type == "reduction_dbstream_wednesday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dbstream = True
        args.detection = True

        target_nodes = ["192.168.10.50"]
        # reduce_helper("./datasets/cic-ids-2017/Wednesday-WorkingHours.csv", target_nodes, args)    
        merged_edges, _ = prep_data("./datasets/cic-ids-2017/Wednesday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/cic-ids-2017/Wednesday-WorkingHours.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_dpgmm_friday_cic_ids":

        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = True
        args.K = 100

        target_nodes = ["192.168.10.50"]

        merged_edges, _ = prep_data("./datasets/cic-ids-2017/Friday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/cic-ids-2017/Friday-WorkingHours.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_kde_friday_cic_ids":
        args.kernel = "gaussian"
        args.sys_call = True
        args.zero_center = False
        args.detection = True

        target_nodes = ["192.168.10.50"]
        reduce_helper("./datasets/cic-ids-2017/Friday-WorkingHours.csv", target_nodes, args)

    elif args.experiment_type == "reduction_kmeans_friday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_kmeans = True
        args.detection = True

        target_nodes = ["192.168.10.50"]
        merged_edges, _ = prep_data("./datasets/cic-ids-2017/Friday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/cic-ids-2017/Friday-WorkingHours.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_dbstream_friday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dbstream = True
        args.detection = True

        target_nodes = ["192.168.10.50"]
        # reduce_helper("./datasets/cic-ids-2017/Friday-WorkingHours.csv", target_nodes, args)
        merged_edges, _ = prep_data("./datasets/cic-ids-2017/Friday-WorkingHours.csv", target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("./datasets/cic-ids-2017/Friday-WorkingHours.csv", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "reduction_dpgmm_all_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = True
        args.visualize = True

        target_nodes = ["192.168.10.50"]
        csv_path = ["./datasets/cic-ids-2017/Monday-WorkingHours.csv",
                    "./datasets/cic-ids-2017/Tuesday-WorkingHours.csv",
                    "./datasets/cic-ids-2017/Wednesday-WorkingHours.csv",
                    "./datasets/cic-ids-2017/Friday-WorkingHours.csv"]

        merged_edges, _ = prep_data(csv_path, target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        evaluate_enhancement("all_cic_ids", enhanced_edges, merged_edges, args)


    elif args.experiment_type == "update_e5_trace":

        csv_path = "./datasets/E5_Trace/*.csv"        
        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = False

        merged_edges, _ = prep_data(csv_path, None, args)
        # Print out the edge with the most timestamps
        max_edge = max(merged_edges.items(), key=lambda x: len(x[1]))
        max_count = len(max_edge[1])
        print("Count of timestamps for the edge with the most timestamps:", max_count, flush=True)        
        splits = split_data(merged_edges)
        enhanced_edges = process_splits(splits, args)
        evaluate_enhancement("e5-trace-split-2", enhanced_edges, merged_edges, args)

    elif args.experiment_type == "update_e5_cadets":
        csv_path = "./datasets/E5_Cadets/*.csv"
        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = False

        merged_edges, _ = prep_data(csv_path, None, args)

        max_edge = max(merged_edges.items(), key=lambda x: len(x[1]))
        max_count = len(max_edge[1])
        print("Count of timestamps for the edge with the most timestamps:", max_count, flush=True)  

        splits = split_data(merged_edges)
        enhanced_edges = process_splits(splits, args)
        evaluate_enhancement(csv_path, enhanced_edges, merged_edges, args)

    elif args.experiment_type == "update_monday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = True

        target_nodes = ["192.168.10.50"]

        csv_path = "./datasets/cic-ids-2017/Monday-WorkingHours.csv"
        merged_edges, scalers = prep_data(csv_path, target_nodes, args)

        

        splits = split_data(merged_edges)
        enhanced_edges = process_splits(splits, args)
        print("Length of enhanced edges:", len(enhanced_edges), flush=True)
        evaluate_enhancement(csv_path, enhanced_edges, merged_edges, args)

        key = list(enhanced_edges.keys())[0]
        timestamps = merged_edges[key]
        scaler = scalers[0]

        enhanced_edges[key].visualize_distribution(name="monday_temporal_pattern_update", timestamps=timestamps, scaler=scaler)

    elif args.experiment_type == "update_tuesday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = True
        # args.visualize = True

        target_nodes = ["192.168.10.50"]

        csv_path = "./datasets/cic-ids-2017/Tuesday-WorkingHours.csv"
        merged_edges, scalers = prep_data(csv_path, target_nodes, args)

        splits = split_data(merged_edges)
        enhanced_edges = process_splits(splits, args)
        evaluate_enhancement("tuesday-split-1", enhanced_edges, merged_edges, args)

        key = list(enhanced_edges.keys())[0]
        timestamps = merged_edges[key]
        scaler = scalers[0]

        enhanced_edges[key].visualize_distribution(name="tuesday_temporal_pattern_update", timestamps=timestamps, scaler=scaler)


    elif args.experiment_type == "update_wednesday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = True

        target_nodes = ["192.168.10.50"]

        csv_path = "./datasets/cic-ids-2017/Wednesday-WorkingHours.csv"
        merged_edges, _ = prep_data(csv_path, target_nodes, args)

        splits = split_data(merged_edges)
        enhanced_edges = process_splits(splits, args)
        evaluate_enhancement(csv_path, enhanced_edges, merged_edges, args)

    elif args.experiment_type == "update_friday_cic_ids":
        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = True

        target_nodes = ["192.168.10.50"]

        csv_path = "./datasets/cic-ids-2017/Friday-WorkingHours.csv"
        merged_edges, _ = prep_data(csv_path, target_nodes, args)

        splits = split_data(merged_edges)
        enhanced_edges = process_splits(splits, args)
        evaluate_enhancement(csv_path, enhanced_edges, merged_edges, args)

    elif args.experiment_type == "update_cic_ids":
        args.kernel_type = "gaussian"
        args.sys_call = True
        args.zero_center = False
        args.use_dpgmm = True
        args.detection = True
        args.visualize = True
        target_nodes = ["192.168.10.50"]

        # csv_path = "./datasets/cic-ids-2017/*.csv"
        csv_path = ["./datasets/cic-ids-2017/Monday-WorkingHours.csv",
            "./datasets/cic-ids-2017/Tuesday-WorkingHours.csv",
            "./datasets/cic-ids-2017/Wednesday-WorkingHours.csv",
            "./datasets/cic-ids-2017/Friday-WorkingHours.csv"]
        merged_edges, scalers = prep_data(csv_path, target_nodes, args)
        assert len(merged_edges) == 1
        print("before splitting: ", merged_edges, flush=True)
        splits = split_data_cic_ids(merged_edges, scalers[0], args)


        # enhanced_edges = process_splits([splits[0], splits[1]], args)
        evaluation_edges = merge_two_merged_edges(splits[0], splits[1])
        # evaluate_enhancement("cic_ids_1_split", enhanced_edges, evaluation_edges, args)

        # enhanced_edges = process_splits([splits[0], splits[1], splits[2]], args)
        evaluation_edges = merge_two_merged_edges(evaluation_edges, splits[2])
        # evaluate_enhancement("cic_ids_2_split", enhanced_edges, evaluation_edges, args)

        enhanced_edges = process_splits([splits[0], splits[1], splits[2], splits[3]], args)
        evaluation_edges = merge_two_merged_edges(evaluation_edges, splits[3])
        evaluate_enhancement("cic_ids_3_split", enhanced_edges, evaluation_edges, args)


    elif args.experiment_type == "attack_simulation":
        args.kernel = "gaussian"
        args.sys_call = True
        args.detection = True

        target_nodes = ["192.168.10.50"]

        merged_edges, scalers = prep_data("./datasets/cic-ids-2017/Monday-WorkingHours.csv", target_nodes, args)
        enhanced_edges_monday = fit_data(merged_edges, args)
        # evaluate_enhancement("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", enhanced_edges, merged_edges, args)
        # enhanced_edges.visualize_distribution(name="attack_example_tuesday")
        key = list(enhanced_edges_monday.keys())[0]
        timestamps = merged_edges[key]
        scaler = scalers[0]

        enhanced_edges_monday[key].visualize_distribution(name="monday_temporal_pattern", timestamps=timestamps, scaler=scaler)
    
        args.kernel = "gaussian"
        args.sys_call = True
        args.detection = True

        target_nodes = ["192.168.10.50"]

        merged_edges, scalers = prep_data("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", target_nodes, args)
        enhanced_edges_tuesday = fit_data(merged_edges, args)
        # evaluate_enhancement("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", enhanced_edges, merged_edges, args)
        # enhanced_edges.visualize_distribution(name="attack_example_tuesday")
        key = list(enhanced_edges_tuesday.keys())[0]
        timestamps = merged_edges[key]
        scaler = scalers[0]

        enhanced_edges_tuesday[key].visualize_distribution(name="tuesday_temporal_pattern", timestamps=timestamps, scaler=scaler)

        ks_statistics, p_value = ks_test_kdes(enhanced_edges_tuesday[key].kde, enhanced_edges_monday[key].kde)
        print("KS Statistic: ", ks_statistics, flush=True)
        print("P-value: ", p_value, flush=True)
        
        
        args.kernel = "gaussian"
        args.sys_call = True
        args.detection = True

        target_nodes = ["192.168.10.50"]

        csv_paths = ["./datasets/cic-ids-2017/Monday-WorkingHours.csv", "./datasets/cic-ids-2017/Tuesday-WorkingHours.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        # evaluate_enhancement("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv", enhanced_edges, merged_edges, args)
        # enhanced_edges.visualize_distribution(name="attack_example_tuesday")
        key = list(enhanced_edges.keys())[0]
        timestamps = merged_edges[key]
        scaler = scalers[0]

        enhanced_edges[key].visualize_distribution(name="combined_temporal_pattern", timestamps=timestamps, scaler=scaler)

    elif args.experiment_type == "create_static_detection_dataset":
        target_nodes = ["192.168.10.50"]
        synthetic_dfs = [pd.read_csv(f"./datasets/cic-ids-2017/synthetic-fourth/noise_{i}.csv") for i in range(6)]


        real_df = pd.read_csv("./datasets/cic-ids-2017/Monday-WorkingHours.csv")
        real_df = real_df[real_df['objectname'].isin(target_nodes)]

        dfs = synthetic_dfs + [real_df]
        offset = 0
        for df in dfs:
            seconds_to_add = offset * 24 * 60 * 60
            df['timestamp'] = df['timestamp'] + seconds_to_add
            df.to_csv(f'./datasets/cic-ids-2017/static-detection/{offset}.csv', index=False)
            offset += 1

        synthetic_dfs = [pd.read_csv(f"./datasets/cic-ids-2017/synthetic-fourth/noise_{i}.csv") for i in range(6)]
        real_df = pd.read_csv("./datasets/cic-ids-2017/Tuesday-WorkingHours.csv")
        real_df = real_df[real_df['objectname'].isin(target_nodes)]
        real_df['timestamp'] = real_df['timestamp'] - (1 * 24 * 60 * 60)

        dfs = synthetic_dfs + [real_df]

        for df in dfs:
            seconds_to_add = offset * 24 * 60 * 60
            df['timestamp'] = df['timestamp'] + seconds_to_add
            df.to_csv(f'./datasets/cic-ids-2017/static-detection/{offset}.csv', index=False)
            offset += 1

        synthetic_dfs = [pd.read_csv(f"./datasets/cic-ids-2017/synthetic-fourth/noise_{i}.csv") for i in range(6)]
        real_df = pd.read_csv("./datasets/cic-ids-2017/Wednesday-WorkingHours.csv")
        real_df = real_df[real_df['objectname'].isin(target_nodes)]
        real_df['timestamp'] = real_df['timestamp'] - (2 * 24 * 60 * 60)

        dfs = synthetic_dfs + [real_df]
        for df in dfs:
            seconds_to_add = offset * 24 * 60 * 60
            df['timestamp'] = df['timestamp'] + seconds_to_add
            df.to_csv(f'./datasets/cic-ids-2017/static-detection/{offset}.csv', index=False)
            offset += 1
        

        synthetic_dfs = [pd.read_csv(f"./datasets/cic-ids-2017/synthetic-fourth/noise_{i}.csv") for i in range(6)]
        real_df = pd.read_csv("./datasets/cic-ids-2017/Friday-WorkingHours.csv")
        real_df = real_df[real_df['objectname'].isin(target_nodes)]
        real_df['timestamp'] = real_df['timestamp'] - (4 * 24 * 60 * 60)

        dfs = synthetic_dfs + [real_df]
        for df in dfs:
            seconds_to_add = offset * 24 * 60 * 60
            df['timestamp'] = df['timestamp'] + seconds_to_add
            df.to_csv(f'./datasets/cic-ids-2017/static-detection/{offset}.csv', index=False)
            offset += 1

        synthetic_dfs = [pd.read_csv(f"./datasets/cic-ids-2017/synthetic-fourth/noise_{i}.csv") for i in range(6)]
        dfs = synthetic_dfs
        for df in dfs:
            seconds_to_add = offset * 24 * 60 * 60
            df['timestamp'] = df['timestamp'] + seconds_to_add
            df.to_csv(f'./datasets/cic-ids-2017/static-detection/{offset}.csv', index=False)
            offset += 1

    elif args.experiment_type == "static_detection_tuesday":
        print("Running static detection tuesday", flush=True)

        args.static_detection = True
        # args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "atlas"
        # args.K = 100

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 14 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(28, 34)]       
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        
        minmax_scaler = MinMaxScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = minmax_scaler.fit_transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13, 14)]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)


        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = minmax_scaler.transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)

        args.static_detection = True
        # args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "atlas_count"
        # args.K = 100

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 7 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(28, 34)]       

        # No need for target nodes since it is already pre selected for the static detection dataset
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        
        std_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        # constant = 2.0  # Example constant
        # edge_features[:, 2] *= constant
        print("Edge features: ", edge_features, flush=True)

        timestamps = edge_features[:, :2]
        other_features = edge_features[:, 2:]

        timestamps = minmax_scaler.fit_transform(timestamps)
        other_features = std_scaler.fit_transform(other_features)

        edge_features = np.hstack((timestamps, other_features))
        edge_features = torch.tensor(edge_features, dtype=torch.float32)

        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13, 14)]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        
        timestamps = edge_features[:, :2]
        other_features = edge_features[:, 2:]
        
        timestamps = minmax_scaler.transform(timestamps)
        other_features = std_scaler.transform(other_features)

        edge_features = np.hstack((timestamps, other_features))
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)


        args.static_detection = True
        args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "kde"
        args.K = 10

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 7 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(28, 34)]       

        # No need for target nodes since it is already pre selected for the static detection dataset
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        
        std_scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, enhanced_edges=enhanced_edges, scalers=scalers, args=args)
        # constant = 1000.0  # Example constant
        # edge_features[:, 2] *= constant
        print("Edge features: ", edge_features, flush=True)
        
        timestamps = edge_features[:, :2]
        other_features = edge_features[:, 2:]
        
        timestamps = minmax_scaler.fit_transform(timestamps)
        other_features = std_scaler.fit_transform(other_features)

        edge_features = np.hstack((timestamps, other_features))
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13, 14)]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)

        edge_features = create_edge_features(merged_edges=merged_edges, enhanced_edges=enhanced_edges, scalers=scalers, args=args)
        timestamps = edge_features[:, :2]
        other_features = edge_features[:, 2:]
        
        timestamps = minmax_scaler.transform(timestamps)
        other_features = std_scaler.transform(other_features)

        edge_features = np.hstack((timestamps, other_features))
        edge_features = torch.tensor(edge_features, dtype=torch.float32)    
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)
        

    elif args.experiment_type == "static_detection_wednesday":
        
        args.static_detection = True
        # args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "atlas"
        # args.K = 100

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 7 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]        # No need for target nodes since it is already pre selected for the static detection dataset
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        
        std_scaler = StandardScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = std_scaler.fit_transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{20}.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = std_scaler.transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)

        args.static_detection = True
        # args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "atlas_count"
        # args.K = 100

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 7 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]        # No need for target nodes since it is already pre selected for the static detection dataset
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        
        std_scaler = StandardScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        # constant = 1000.0  # Example constant
        # edge_features[:, 2] *= constant
        print("Edge features: ", edge_features, flush=True)
        
        edge_features = std_scaler.fit_transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{20}.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = std_scaler.transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)


        args.static_detection = True
        args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "kde"
        args.K = 100

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 7 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]        # No need for target nodes since it is already pre selected for the static detection dataset
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        
        std_scaler = StandardScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, enhanced_edges=enhanced_edges, scalers=scalers, args=args)
        # constant = 1000.0  # Example constant
        # edge_features[:, 2] *= constant
        print("Edge features: ", edge_features, flush=True)
        
        edge_features = std_scaler.fit_transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{20}.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)

        edge_features = create_edge_features(merged_edges=merged_edges, enhanced_edges=enhanced_edges, scalers=scalers, args=args)
        edge_features = std_scaler.transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)


    elif args.experiment_type == "static_detection_friday":
        args.static_detection = True
        # args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "atlas"
        # args.K = 100

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 7 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]        # No need for target nodes since it is already pre selected for the static detection dataset
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(28, 34)]       
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        
        std_scaler = StandardScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = std_scaler.fit_transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{27}.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = std_scaler.transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)

        args.static_detection = True
        # args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "atlas_count"
        # args.K = 100

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 7 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]        # No need for target nodes since it is already pre selected for the static detection dataset
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(28, 34)]       

        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        
        std_scaler = StandardScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        # constant = 1000.0  # Example constant
        # edge_features[:, 2] *= constant
        print("Edge features: ", edge_features, flush=True)
        
        edge_features = std_scaler.fit_transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{27}.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = std_scaler.transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)


        args.static_detection = True
        args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "kde"
        args.K = 100

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 7 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]        # No need for target nodes since it is already pre selected for the static detection dataset
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(28, 34)]       

        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)
        
        std_scaler = StandardScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, enhanced_edges=enhanced_edges, scalers=scalers, args=args)
        # constant = 1000.0  # Example constant
        # edge_features[:, 2] *= constant
        print("Edge features: ", edge_features, flush=True)
        
        edge_features = std_scaler.fit_transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{27}.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        enhanced_edges = fit_data(merged_edges, args)

        edge_features = create_edge_features(merged_edges=merged_edges, enhanced_edges=enhanced_edges, scalers=scalers, args=args)
        edge_features = std_scaler.transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)

    elif args.experiment_type == "adpative_adversarial_tuesday":
        print("Running static detection tuesday", flush=True)

        args.static_detection = True
        # args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "atlas"
        # args.K = 100

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 14 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(28, 34)]       
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        
        minmax_scaler = MinMaxScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = minmax_scaler.fit_transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13, 14)]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)


        # Target start and end times (human-readable format)
        target_start_time = "2017-07-16 08:16:16"
        target_end_time = "2017-07-16 15:57:37"

        set_adversarial_edge_timestamps(target_start_time=target_start_time,
                                        target_end_time=target_end_time,
                                        merged_edges=merged_edges,
                                        scalers=scalers)


        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = minmax_scaler.transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)

    elif args.experiment_type == "adpative_adversarial_wednesday":
        args.static_detection = True
        # args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "atlas"
        # args.K = 100

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 7 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]        # No need for target nodes since it is already pre selected for the static detection dataset
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        std_scaler = StandardScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = std_scaler.fit_transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{20}.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        target_start_time = "2017-07-23 07:46:45"
        target_end_time = "2017-07-23 15:59:17"

        set_adversarial_edge_timestamps(target_start_time=target_start_time,
                                target_end_time=target_end_time,
                                merged_edges=merged_edges,
                                scalers=scalers)


        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = std_scaler.transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)

    elif args.experiment_type == "adpative_adversarial_friday":
        args.static_detection = True
        # args.use_dpgmm = True
        args.kernel = "gaussian"
        args.distribution = "atlas"

        target_nodes = ["192.168.10.50"]
        attacker_victim_pairs = [("172.16.0.1_same", "192.168.10.50_same")]

        # The first 7 csvs are benign
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(13)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(14, 20)]
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(21, 27)]        # No need for target nodes since it is already pre selected for the static detection dataset
        csv_paths += [f"./datasets/cic-ids-2017/static-detection/{i}.csv" for i in range(28, 34)]       
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)
        
        std_scaler = StandardScaler()
        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = std_scaler.fit_transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges, [])
        
        model = train_baseline(data=edge_features, y=y, args=args)

        
        csv_paths = [f"./datasets/cic-ids-2017/static-detection/{27}.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        # for (key, timestamps), scaler in zip(merged_edges.items(), scalers):
        #     # Inverse transform the scaled timestamps to get the original timestamps
        #     original_timestamps = scaler.inverse_transform(np.array(timestamps).reshape(-1, 1)).flatten()
        #     # For all other edges, just print the human-readable timestamps
        #     start_time_human = datetime.fromtimestamp(min(original_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
        #     end_time_human = datetime.fromtimestamp(max(original_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
        #     print(f"Edge: {key}, Start Time: {start_time_human}, End Time: {end_time_human}", flush=True)

        target_start_time = "2017-07-30 08:05:41"
        target_end_time = "2017-07-30 16:02:40"

        set_adversarial_edge_timestamps(target_start_time=target_start_time,
                                target_end_time=target_end_time,
                                merged_edges=merged_edges,
                                scalers=scalers)

        edge_features = create_edge_features(merged_edges=merged_edges, scalers=scalers, args=args)
        edge_features = std_scaler.transform(edge_features)
        edge_features = torch.tensor(edge_features, dtype=torch.float32)
        y = create_true_labels(merged_edges=merged_edges, attacker_victim_pairs=attacker_victim_pairs)

        evaluate_autoencoder(model=model, data=edge_features, y=y, merged_edges=merged_edges)


    elif args.experiment_type == "online_detection_tuesday":
        print("Starting online detection FTP Parator", flush=True)
        args.detection = True
        args.use_dpgmm = True
        args.window_size = 60
        args.martingale_window_size = 10
        args.use_density = True
        
        target_nodes = ["192.168.10.50"]

        csv_paths = ["./datasets/cic-ids-2017/Monday-WorkingHours.csv", "./datasets/cic-ids-2017/Tuesday-WorkingHours.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        scaler = scalers[0]
        monday_end_time = scaler.transform(np.array([[1499140799]]))
        tuesday_start_time = scaler.transform(np.array([[1499140800]]))
        monday_merged_edges, tuesday_merged_edges = separate_two_merged_edges(merged_edges, monday_end_time, tuesday_start_time)

        assert len(monday_merged_edges) == 1
        assert len(tuesday_merged_edges) == 1

        # enhanced_edges = fit_data(monday_merged_edges, args)
        # with open("./enhanced_edges.pkl", "wb") as f:
        #     pickle.dump(enhanced_edges, f)
        with open("./enhanced_edges.pkl", "rb") as f:
            enhanced_edges = pickle.load(f) 
        key = list(enhanced_edges.keys())[0]
        monday_timestamps = monday_merged_edges[key]
        monday_timestamps = scaler.inverse_transform(monday_timestamps)
        monday_enhanced_edge = enhanced_edges[key]

        
        recency_detector, curr_window_timestamps, prev_features = train_recency_detector(timestamps=monday_timestamps, enhanced_edge=monday_enhanced_edge, scaler=scaler, args=args)
        # print("current window timestamps: ", curr_window_timestamps, flush=True)
        print("Prev features shape: ", prev_features.shape, flush=True)
        print("prev features: ", prev_features, flush=True)
    

        tuesday_timestamps = tuesday_merged_edges[key]
        tuesday_timestamps = scaler.inverse_transform(tuesday_timestamps)
        tuesday_timestamps = tuesday_timestamps[(tuesday_timestamps > 1499170500) & (tuesday_timestamps < 1499174400)]  # Filter timestamps after 1499170500
        args.use_dpgmm = False
        detect_recency_detector(recency_detector, tuesday_timestamps, prev_features, args)
        recency_detector.enhanced_edge.visualize_distribution(tuesday_merged_edges[key], name="recency_classifier_check")
        
        print("Starting online detection SSH Parator", flush=True)

        args.detection = True
        args.use_dpgmm = True
        args.window_size = 60
        args.martingale_window_size = 10
        args.use_density = True
        
        target_nodes = ["192.168.10.50"]

        csv_paths = ["./datasets/cic-ids-2017/Monday-WorkingHours.csv", "./datasets/cic-ids-2017/Tuesday-WorkingHours.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        scaler = scalers[0]
        monday_end_time = scaler.transform(np.array([[1499140799]]))
        tuesday_start_time = scaler.transform(np.array([[1499140800]]))
        monday_merged_edges, tuesday_merged_edges = separate_two_merged_edges(merged_edges, monday_end_time, tuesday_start_time)

        assert len(monday_merged_edges) == 1
        assert len(tuesday_merged_edges) == 1

        # enhanced_edges = fit_data(monday_merged_edges, args)
        with open("./enhanced_edges.pkl", "rb") as f:
            enhanced_edges = pickle.load(f) 
        key = list(enhanced_edges.keys())[0]
        monday_timestamps = monday_merged_edges[key]
        monday_timestamps = scaler.inverse_transform(monday_timestamps)
        monday_enhanced_edge = enhanced_edges[key]
        
        recency_detector, curr_window_timestamps, prev_features = train_recency_detector(timestamps=monday_timestamps, enhanced_edge=monday_enhanced_edge, scaler=scaler, args=args)
        # print("current window timestamps: ", curr_window_timestamps, flush=True)
        print("Length of current window timestamps: ", len(curr_window_timestamps), flush=True)
    

        tuesday_timestamps = tuesday_merged_edges[key]
        tuesday_timestamps = scaler.inverse_transform(tuesday_timestamps)
        tuesday_timestamps = tuesday_timestamps[(tuesday_timestamps > 1499187300) & (tuesday_timestamps < 1499191800)]  # Filter timestamps after 1499170500
        args.use_dpgmm = False
        detect_recency_detector(recency_detector, tuesday_timestamps, prev_features, args)

    elif args.experiment_type == "online_detection_wednesday":
        print("Starting online detection DoS slowloris", flush=True)
        args.detection = True
        args.use_dpgmm = True
        args.window_size = 60
        args.martingale_window_size = 10
        args.use_density = True
        
        target_nodes = ["192.168.10.50"]

        csv_paths = ["./datasets/cic-ids-2017/Monday-WorkingHours.csv", "./datasets/cic-ids-2017/Wednesday-WorkingHours.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        scaler = scalers[0]
        monday_end_time = scaler.transform(np.array([[1499140799]]))
        wednesday_start_time = scaler.transform(np.array([[1499227200]]))
        print("Wednesday human readable start time: ", datetime.fromtimestamp(wednesday_start_time[0][0]).strftime('%Y-%m-%d %H:%M:%S'), flush=True)
        monday_merged_edges, wednesday_merged_edges = separate_two_merged_edges(merged_edges, monday_end_time, wednesday_start_time)

        assert len(monday_merged_edges) == 1
        assert len(wednesday_merged_edges) == 1

        # enhanced_edges = fit_data(monday_merged_edges, args)
        # with open("./enhanced_edges.pkl", "wb") as f:
        #     pickle.dump(enhanced_edges, f)
        with open("./enhanced_edges.pkl", "rb") as f:
            enhanced_edges = pickle.load(f) 
        key = list(enhanced_edges.keys())[0]
        monday_timestamps = monday_merged_edges[key]
        print("Length of monday timestamps: ", len(monday_timestamps), flush=True)
        monday_timestamps = scaler.inverse_transform(monday_timestamps)
        monday_enhanced_edge = enhanced_edges[key]

        
        recency_detector, curr_window_timestamps, prev_features = train_recency_detector(timestamps=monday_timestamps, enhanced_edge=monday_enhanced_edge, scaler=scaler, args=args)
        # print("current window timestamps: ", curr_window_timestamps, flush=True)
        print("Prev features shape: ", prev_features.shape, flush=True)
        print("prev features: ", prev_features, flush=True)
    

        wednesday_timestamps = wednesday_merged_edges[key]
        print("Length of wednesday timestamps: ", len(wednesday_timestamps), flush=True)
        wednesday_timestamps = scaler.inverse_transform(wednesday_timestamps)
        wednesday_timestamps = wednesday_timestamps[(wednesday_timestamps > 1499258520) & (wednesday_timestamps < 1499260440)]  # Filter timestamps after 1499170500
        args.use_dpgmm = False
        detect_recency_detector(recency_detector, wednesday_timestamps, prev_features, args)

        print("Starting online detection DoS Slowhttptest", flush=True)
        args.detection = True
        args.use_dpgmm = True
        args.window_size = 60
        args.martingale_window_size = 10
        args.use_density = True
        
        target_nodes = ["192.168.10.50"]

        csv_paths = ["./datasets/cic-ids-2017/Monday-WorkingHours.csv", "./datasets/cic-ids-2017/Wednesday-WorkingHours.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        scaler = scalers[0]
        monday_end_time = scaler.transform(np.array([[1499140799]]))
        wednesday_start_time = scaler.transform(np.array([[1499227200]]))
        print("Wednesday human readable start time: ", datetime.fromtimestamp(wednesday_start_time[0][0]).strftime('%Y-%m-%d %H:%M:%S'), flush=True)
        monday_merged_edges, wednesday_merged_edges = separate_two_merged_edges(merged_edges, monday_end_time, wednesday_start_time)

        assert len(monday_merged_edges) == 1
        assert len(wednesday_merged_edges) == 1

        # enhanced_edges = fit_data(monday_merged_edges, args)
        # with open("./enhanced_edges.pkl", "wb") as f:
        #     pickle.dump(enhanced_edges, f)
        with open("./enhanced_edges.pkl", "rb") as f:
            enhanced_edges = pickle.load(f) 
        key = list(enhanced_edges.keys())[0]
        monday_timestamps = monday_merged_edges[key]
        print("Length of monday timestamps: ", len(monday_timestamps), flush=True)
        monday_timestamps = scaler.inverse_transform(monday_timestamps)
        monday_enhanced_edge = enhanced_edges[key]

        
        recency_detector, curr_window_timestamps, prev_features = train_recency_detector(timestamps=monday_timestamps, enhanced_edge=monday_enhanced_edge, scaler=scaler, args=args)
        # print("current window timestamps: ", curr_window_timestamps, flush=True)
        print("Prev features shape: ", prev_features.shape, flush=True)
        print("prev features: ", prev_features, flush=True)
    

        wednesday_timestamps = wednesday_merged_edges[key]
        print("Length of wednesday timestamps: ", len(wednesday_timestamps), flush=True)
        wednesday_timestamps = scaler.inverse_transform(wednesday_timestamps)
        wednesday_timestamps = wednesday_timestamps[(wednesday_timestamps > 1499260200) & (wednesday_timestamps < 1499262180)]  # Filter timestamps after 1499170500
        args.use_dpgmm = False
        detect_recency_detector(recency_detector, wednesday_timestamps, prev_features, args)

        print("Starting online detection DoS Hulk", flush=True)
        args.detection = True
        args.use_dpgmm = True
        args.window_size = 60
        args.martingale_window_size = 10
        args.use_density = True
        
        target_nodes = ["192.168.10.50"]

        csv_paths = ["./datasets/cic-ids-2017/Monday-WorkingHours.csv", "./datasets/cic-ids-2017/Wednesday-WorkingHours.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        scaler = scalers[0]
        monday_end_time = scaler.transform(np.array([[1499140799]]))
        wednesday_start_time = scaler.transform(np.array([[1499227200]]))
        print("Wednesday human readable start time: ", datetime.fromtimestamp(wednesday_start_time[0][0]).strftime('%Y-%m-%d %H:%M:%S'), flush=True)
        monday_merged_edges, wednesday_merged_edges = separate_two_merged_edges(merged_edges, monday_end_time, wednesday_start_time)

        assert len(monday_merged_edges) == 1
        assert len(wednesday_merged_edges) == 1

        # enhanced_edges = fit_data(monday_merged_edges, args)
        # with open("./enhanced_edges.pkl", "wb") as f:
        #     pickle.dump(enhanced_edges, f)
        with open("./enhanced_edges.pkl", "rb") as f:
            enhanced_edges = pickle.load(f) 
        key = list(enhanced_edges.keys())[0]
        monday_timestamps = monday_merged_edges[key]
        print("Length of monday timestamps: ", len(monday_timestamps), flush=True)
        monday_timestamps = scaler.inverse_transform(monday_timestamps)
        monday_enhanced_edge = enhanced_edges[key]

        
        recency_detector, curr_window_timestamps, prev_features = train_recency_detector(timestamps=monday_timestamps, enhanced_edge=monday_enhanced_edge, scaler=scaler, args=args)
        # print("current window timestamps: ", curr_window_timestamps, flush=True)
        print("Prev features shape: ", prev_features.shape, flush=True)
        print("prev features: ", prev_features, flush=True)
    

        wednesday_timestamps = wednesday_merged_edges[key]
        print("Length of wednesday timestamps: ", len(wednesday_timestamps), flush=True)
        wednesday_timestamps = scaler.inverse_transform(wednesday_timestamps)

        wednesday_timestamps = wednesday_timestamps[(wednesday_timestamps > 1499261880) & (wednesday_timestamps < 1499263200)]  # Filter timestamps after 1499170500
        args.use_dpgmm = False
        detect_recency_detector(recency_detector, wednesday_timestamps, prev_features, args)



        print("Starting online detection DoS GoldenEye", flush=True)
        args.detection = True
        args.use_dpgmm = True
        args.window_size = 60
        args.martingale_window_size = 10
        args.use_density = True
        
        target_nodes = ["192.168.10.50"]

        csv_paths = ["./datasets/cic-ids-2017/Monday-WorkingHours.csv", "./datasets/cic-ids-2017/Wednesday-WorkingHours.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        scaler = scalers[0]
        monday_end_time = scaler.transform(np.array([[1499140799]]))
        wednesday_start_time = scaler.transform(np.array([[1499227200]]))
        print("Wednesday human readable start time: ", datetime.fromtimestamp(wednesday_start_time[0][0]).strftime('%Y-%m-%d %H:%M:%S'), flush=True)
        monday_merged_edges, wednesday_merged_edges = separate_two_merged_edges(merged_edges, monday_end_time, wednesday_start_time)

        assert len(monday_merged_edges) == 1
        assert len(wednesday_merged_edges) == 1

        # enhanced_edges = fit_data(monday_merged_edges, args)
        # with open("./enhanced_edges.pkl", "wb") as f:
        #     pickle.dump(enhanced_edges, f)
        with open("./enhanced_edges.pkl", "rb") as f:
            enhanced_edges = pickle.load(f) 
        key = list(enhanced_edges.keys())[0]
        monday_timestamps = monday_merged_edges[key]
        print("Length of monday timestamps: ", len(monday_timestamps), flush=True)
        monday_timestamps = scaler.inverse_transform(monday_timestamps)
        monday_enhanced_edge = enhanced_edges[key]

        
        recency_detector, curr_window_timestamps, prev_features = train_recency_detector(timestamps=monday_timestamps, enhanced_edge=monday_enhanced_edge, scaler=scaler, args=args)
        # print("current window timestamps: ", curr_window_timestamps, flush=True)
        print("Prev features shape: ", prev_features.shape, flush=True)
        print("prev features: ", prev_features, flush=True)
    

        wednesday_timestamps = wednesday_merged_edges[key]
        print("Length of wednesday timestamps: ", len(wednesday_timestamps), flush=True)
        wednesday_timestamps = scaler.inverse_transform(wednesday_timestamps)
        # df = df[(df['timestamp'] > 1499263200) & (df['timestamp'] < 1499265000)]

        wednesday_timestamps = wednesday_timestamps[(wednesday_timestamps > 1499263200) & (wednesday_timestamps < 1499265000)]  # Filter timestamps after 1499170500
        args.use_dpgmm = False
        detect_recency_detector(recency_detector, wednesday_timestamps, prev_features, args)

    elif args.experiment_type == "online_detection_friday":
        print("Starting online detection DDoS LOIT", flush=True)
        args.detection = True
        args.use_dpgmm = True
        args.window_size = 60
        args.martingale_window_size = 10
        args.use_density = True
        
        target_nodes = ["192.168.10.50"]

        csv_paths = ["./datasets/cic-ids-2017/Monday-WorkingHours.csv", "./datasets/cic-ids-2017/Friday-WorkingHours.csv"]
        merged_edges, scalers = prep_data(csv_paths, target_nodes, args)

        scaler = scalers[0]
        monday_end_time = scaler.transform(np.array([[1499140799]]))
        friday_start_time = scaler.transform(np.array([[1499400000]]))
        print("Wednesday human readable start time: ", datetime.fromtimestamp(friday_start_time[0][0]).strftime('%Y-%m-%d %H:%M:%S'), flush=True)
        monday_merged_edges, friday_merged_edges = separate_two_merged_edges(merged_edges, monday_end_time, friday_start_time)

        assert len(monday_merged_edges) == 1
        assert len(friday_merged_edges) == 1

        # enhanced_edges = fit_data(monday_merged_edges, args)
        # with open("./enhanced_edges.pkl", "wb") as f:
        #     pickle.dump(enhanced_edges, f)
        with open("./enhanced_edges.pkl", "rb") as f:
            enhanced_edges = pickle.load(f) 
        key = list(enhanced_edges.keys())[0]
        monday_timestamps = monday_merged_edges[key]
        print("Length of monday timestamps: ", len(monday_timestamps), flush=True)
        monday_timestamps = scaler.inverse_transform(monday_timestamps)
        monday_enhanced_edge = enhanced_edges[key]

        
        recency_detector, curr_window_timestamps, prev_features = train_recency_detector(timestamps=monday_timestamps, enhanced_edge=monday_enhanced_edge, scaler=scaler, args=args)
        # print("current window timestamps: ", curr_window_timestamps, flush=True)
        print("Prev features shape: ", prev_features.shape, flush=True)
        print("prev features: ", prev_features, flush=True)
    

        friday_timestamps = friday_merged_edges[key]
        print("Length of friday timestamps: ", len(friday_timestamps), flush=True)
        friday_timestamps = scaler.inverse_transform(friday_timestamps)

        friday_timestamps = friday_timestamps[(friday_timestamps > 1499453460) & (friday_timestamps < 1499455200)]  # Filter timestamps after 1499170500
        args.use_dpgmm = False
        detect_recency_detector(recency_detector, friday_timestamps, prev_features, args)








