from scipy.stats import ks_2samp
import numpy as np
from datetime import datetime
from recency_classifier import RecencyDetector
from collections import deque
from tqdm import tqdm 
import torch


# simple KS test with winodwing
def online_detection_KS_test(curr_start_timestamp, curr_end_timestamp, difference, curr_kde, old_kdes):
    print("Detecting using KS test...", flush=True)
    all_different = True
    for old_kde in old_kdes:
        curr_points = np.linspace(curr_start_timestamp, curr_end_timestamp, 1000)[:, np.newaxis]
        curr_density = np.exp(curr_kde.score_samples(curr_points))

        # Convert and print current timestamps in human-readable format
        print("curr start time", curr_start_timestamp, "(", datetime.fromtimestamp(curr_start_timestamp).strftime('%Y-%m-%d %H:%M:%S'), ")", flush=True)
        print("curr end time", curr_end_timestamp, "(", datetime.fromtimestamp(curr_end_timestamp).strftime('%Y-%m-%d %H:%M:%S'), ")", flush=True)

        old_start_timestamp = curr_start_timestamp - difference
        old_end_timestamp = curr_end_timestamp - difference

        # Convert and print old timestamps in human-readable format
        print("old start time", old_start_timestamp, "(", datetime.fromtimestamp(old_start_timestamp).strftime('%Y-%m-%d %H:%M:%S'), ")", flush=True)
        print("old end time", old_end_timestamp, "(", datetime.fromtimestamp(old_end_timestamp).strftime('%Y-%m-%d %H:%M:%S'), ")", flush=True)
    
        old_points = np.linspace(old_start_timestamp, old_end_timestamp, 1000)[:, np.newaxis]
        old_density = np.exp(old_kde.score_samples(old_points))
        ks_statistic, p_value = ks_2samp(curr_density, old_density)

        print(f"K-S statistic: {ks_statistic}", flush=True)
        print(f"P-value: {p_value}", flush=True)
        # print(f"curr p-value: {p_value}", flush=True)


        if p_value > 0.05:
            all_different = False
            print("The curr distribution is not significantly different from the old one.", flush=True)
            break

    if all_different:
        print("The curr distribution is significant different from all others.", flush=True)
        return True
    
    return False


def train_recency_detector(timestamps, enhanced_edge, scaler, args):

    recency_detector = RecencyDetector(enhanced_edge=enhanced_edge, scaler=scaler)

    features, curr_window_timestamps = recency_detector.create_edge_features(timestamps=timestamps, args=args)
    print("Features shape: ", features.shape, flush=True)
    # print("Current window timestamps: ", curr_window_timestamps, flush=True)

    older_features, recent_features = recency_detector.partition_datasets(features)
    print("Older features shape: ", older_features.shape, flush=True)
    print("Recent features shape: ", recent_features.shape, flush=True)
    older_train, older_test, recent_train, recent_test, input_dim = recency_detector.train_test_split(older_features, recent_features)

    train_pairs, train_labels = recency_detector.generate_pairs(older_train, recent_train)
    test_pairs, test_labels = recency_detector.generate_pairs(older_test, recent_test)

    input_dim = len(train_pairs[0][0]) * 2
    num_epochs = recency_detector.init_model(input_dim)
    
    recency_detector.train_model(num_epochs, train_pairs, train_labels, test_pairs, test_labels)
    recency_detector.evaluate_model(test_pairs, test_labels)
    
    return recency_detector, curr_window_timestamps, features
    

def detect_recency_detector(recency_detector, new_data_timestamps, prev_features, args):
    print("Length of new data timestamps: ", len(new_data_timestamps), flush=True)
    new_data_features, new_window_timestamps = recency_detector.create_edge_features(timestamps=new_data_timestamps, args=args)
    print("length new windowed timestamps: ", len(new_window_timestamps), flush=True)
    print("new data features: ", new_data_features, flush=True)

    S_n = 0
    Y_k_queue = deque()
    detected_timestamps = []
    length_old_windows = 0
    new_features = []
    for i, new_window_timestamp in tqdm(enumerate(new_window_timestamps), total=len(new_window_timestamps), desc="Processing new data stream..."):
        # length_old_windows += 1
        prev_features, new_feature = recency_detector.update_model(prev_features, new_window_timestamps[:i+1], length_old_windows, args=args)
        new_features.append(new_feature)

        Y_k = recency_detector.detect_distribution_shift(new_feature)
        Y_k_queue.append(Y_k)
        if len(Y_k_queue) > args.martingale_window_size:
            Y_k_queue.popleft()

        S_n = sum(Y_k_queue)
        martingale = recency_detector.exponential_martingale(S_n, len(Y_k_queue), t=1, p=0.5, q=0.5)

        print(f"Data Index {i} : martingale={martingale:.3f}")
        if martingale > 10:
            print("Alert: Distribution shift detected!")
            detected_timestamp = min(new_data_timestamps) + args.window_size * (i + 1)
            print(detected_timestamp, flush=True)
            print("Detected human readable timestamp: ", datetime.fromtimestamp(detected_timestamp).strftime('%Y-%m-%d %H:%M:%S'), flush=True)
            detected_timestamps.append(datetime.fromtimestamp(detected_timestamp).strftime('%Y-%m-%d %H:%M:%S'))
            break

    # debug_features(prev_features, torch.stack(new_features))


def debug_features(prev_features, new_features):
    print("=== DEBUGGING FEATURES ===", flush=True)
    # Compute min and max for prev_features
    prev_min = torch.min(prev_features, dim=0).values
    prev_max = torch.max(prev_features, dim=0).values

    # Compute min and max for new_features
    new_min = torch.min(new_features, dim=0).values
    new_max = torch.max(new_features, dim=0).values

    # Print the results for comparison
    print("Previous Features - Min values:", prev_min)
    print("Previous Features - Max values:", prev_max)
    print("New Features - Min values:", new_min)
    print("New Features - Max values:", new_max)