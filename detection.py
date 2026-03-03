import torch
import torch.nn as nn
import torch.optim as optim
from autoencoder import EdgeAutoencoder
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def train_baseline(data, y, args):
    if args.distribution == "kde":
        hidden_dim = 50
    elif args.distribution == "atlas":
        hidden_dim = 1
    elif args.distribution == "atlas_count":
        hidden_dim = 1

    model = train_autoencoder(data, y, hidden_dim)

    return model

def create_true_labels(merged_edges, attacker_victim_pairs):
    y = []
    for key, edge in merged_edges.items():
        if (key[0], key[1]) in attacker_victim_pairs:
            y.append(1)
        else:
            y.append(0) 
    
    y = np.array(y)
    return y


def create_features_matrix(enhanced_edges, merged_edges, scalers, args):

    edge_attr = create_edge_features(enhanced_edges, merged_edges, scalers, args)
    
    print("edge features shape: ", edge_attr.shape, flush=True)

    return edge_attr

def create_edge_features(merged_edges, scalers, args, enhanced_edges=None):
    edge_features = []
    for index, (key, edge) in enumerate(merged_edges.items()):
        curr_scaler = scalers[index]
        if  args.distribution == "atlas":
            start_timestamp = curr_scaler.inverse_transform(np.array([merged_edges[key][0]]))[0, 0]
            end_timestamp = curr_scaler.inverse_transform(np.array([merged_edges[key][-1]]))[0, 0]

            features = [start_timestamp, end_timestamp]
            edge_features.append(features)

        elif args.distribution == "atlas_count":
            start_timestamp = curr_scaler.inverse_transform(np.array([merged_edges[key][0]]))[0, 0]
            end_timestamp = curr_scaler.inverse_transform(np.array([merged_edges[key][-1]]))[0, 0]
        
            count = len(merged_edges[key])

            features = [start_timestamp, end_timestamp, count]
            print("features: ", features, flush=True)
            edge_features.append(features)

        elif args.distribution == "kde":

            enhanced_edge = enhanced_edges[key]

            start_timestamp = merged_edges[key][0][0]  # Convert to scalar
            end_timestamp = merged_edges[key][-1][0]
            # grid = np.linspace(min(edge.timestamps), max(edge.timestamps), args.num_samples)[:, np.newaxis]
            # sampled_points = np.exp(edge.kde.score_samples(grid))
            grid = np.linspace(start_timestamp, end_timestamp, 100)
            print("grid shape: ", grid.shape, flush=True)
            densities = np.exp(enhanced_edge.kde.score_samples(grid[:, np.newaxis]))
            count = len(merged_edges[key])

            start_timestamp = curr_scaler.inverse_transform(np.array([merged_edges[key][0]]))[0, 0]
            end_timestamp = curr_scaler.inverse_transform(np.array([merged_edges[key][-1]]))[0, 0]
            peak_density = np.max(densities)
        
            # features = [source_index, destination_index, start_timestamp] + sampled_points.flatten().tolist()
            features = [start_timestamp, end_timestamp, count] + densities.flatten().tolist()
            # features = [start_timestamp, end_timestamp, count, peak_density]
            print("features: ", features, flush=True)
            # features = [start_timestamp, end_timestamp, count, max_density, avg_density]
            edge_features.append(features)

    edge_features = torch.tensor(edge_features, dtype=torch.float32)

    return edge_features

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def train_autoencoder(data, y, hidden_dim=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training data shape: ", data.shape, flush=True)
    print("Hiidden dimension: ", hidden_dim, flush=True)
    print("edge dimension: ", data.shape[1], flush=True)
    hidden_dim = hidden_dim
    edge_dim = data.shape[1]

    model = EdgeAutoencoder(edge_dim, hidden_dim)
    # model = model.to(device)
    # data = data.to(device)
    model.apply(init_weights)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data, val_data, train_y, val_y = train_test_split(data, y, test_size=0.2, random_state=42)
    print("Train data shape: ", train_data.shape, flush=True)
    # print("Validation data shape: ", val_data.shape, flush=True)
    # train_data = data.edge_attr

    num_epochs = 10000
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop = 100

    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Training..."):
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}', flush= True)
        # for name, param in model.named_parameters():
            # print(f"Gradient for {name}: {param.grad}")

        # I want to see the reconstruction error go down over all
        # evaluate_autoencoder(data, y, model)


        model.eval()
        with torch.no_grad():
            val_output = model(val_data)
            val_loss = criterion(val_output, val_data).item()
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop:
                print("Early stopping triggered")
                break

    # evaluate_autoencoder(data, y, model)

    return model



def evaluate_autoencoder(data, y, model, merged_edges=None, verbose=True):
    print("Evaluating...", flush=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data = data.to(device)
    # model = model.to(device)

    model.eval()
    with torch.no_grad():
        reconstructed = model(data)

    reconstruction_error = torch.mean((data - reconstructed) ** 2, dim=1).numpy()

    if verbose:
        print("Reconstruction error: ", reconstruction_error, flush=True)
    else:
        print("Reconstruction error: ", np.sum(reconstruction_error), flush=True)
    threshold = np.percentile(reconstruction_error, 95)
    anomalies = reconstruction_error > threshold

    if verbose:
        detected_anomalies_indices = np.where(anomalies == 1)[0]
        true_anomalies_indices = np.where(y == 1)[0]
        
        print("Detected anomalies indices: ", detected_anomalies_indices, flush=True)
        print("True anomalies indices: ", true_anomalies_indices, flush=True)
        print("Detected anomalies values: ", reconstruction_error[detected_anomalies_indices], flush=True)
        print("True anomalies values: ", reconstruction_error[true_anomalies_indices], flush=True)

        if np.all(np.isin(true_anomalies_indices, detected_anomalies_indices)):
            print("Detected all anomalies!", flush=True)
        else:
            print("Some anomalies were missed.", flush=True)

        if verbose:

            edges = list(merged_edges.keys())

            detected_edges_u_v = []
            for detected_anomalies_index in detected_anomalies_indices:
                key = edges[detected_anomalies_index]
                detected_edges_u_v.append((key[0], key[1], len(merged_edges[key])))

            true_anomaly_edges_u_v = []
            for true_anomaly_index in true_anomalies_indices:
                key = edges[true_anomaly_index]
                true_anomaly_edges_u_v.append((key[0], key[1], len(merged_edges[key])))

            print("Detected edges: ", detected_edges_u_v, flush=True)  
            print("True anomaly edges: ", true_anomaly_edges_u_v, flush=True)

                # Sort reconstruction errors in descending order
            sorted_indices = np.argsort(-reconstruction_error)
            sorted_errors = reconstruction_error[sorted_indices]

            # Find the smallest threshold that includes all true anomalies
            threshold = sorted_errors[np.max([np.where(sorted_indices == i)[0][0] for i in true_anomalies_indices])]

            # Count false positives
            false_positives = np.sum((reconstruction_error >= threshold) & (y == 0))

            total_non_anomalous = np.sum(y == 0)
            false_positive_percentage = (false_positives / total_non_anomalous) * 100

            print(f"Threshold to detect all true anomalies: {threshold:.4f}", flush=True)
            print(f"Number of false positives at this threshold: {false_positives}", flush=True)
            print(f"False positive percentage: {false_positive_percentage:.2f}%", flush=True)

            # Calculate the percentile rank of the reconstruction error for true anomalies
            # true_anomaly_percentile_ranks = [percentileofscore(reconstruction_error, reconstruction_error[i], kind='rank') for i in true_anomalies_indices]
            # for i, true_anomaly_index in enumerate(true_anomalies_indices):
            #     print(f"True anomaly edge {edges[true_anomaly_index][0]}-{edges[true_anomaly_index][1]} with reconstruction error {reconstruction_error[true_anomaly_index]:.4f} is in the {true_anomaly_percentile_ranks[i]}th percentile", flush=True)
    
