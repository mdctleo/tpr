import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math

class RecencyDetector():
    def __init__(self, enhanced_edge, scaler):
        self.enhanced_edge = enhanced_edge
        self.scaler = scaler
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.total_train_loss = 0.0
        self.train_n = 0

    def create_edge_features(self, timestamps, args):
        edge_features = []
        window_timestamps = []
        total_density_sum = 0  # To track the sum of densities
        total_windows = 0      # To track the number of windows processed
        total_count_sum = 0
        previous_count = 0
        previous_density = 0
        rolling_densities = []
        rolling_counts = []

        data_min, data_max = np.min(timestamps), np.max(timestamps)  # These are unnormalized timestamps (seconds or nanoseconds)

        # Iterate from data_min to data_max using window size step
        for i in range(int(data_min), int(data_max), int(args.window_size)):
            end = i + args.window_size
            features, previous_count, previous_density, total_density_sum, total_count_sum, total_windows, curr_window_timestamps = self.process_window(
                start=i,
                end=end,
                timestamps=timestamps,
                previous_count=previous_count,
                previous_density=previous_density,
                total_density_sum=total_density_sum,
                total_count_sum=total_count_sum,
                total_windows=total_windows,
                rolling_densities=rolling_densities,
                rolling_counts=rolling_counts,
                args=args
            )
            edge_features.append(features)
            window_timestamps.append(curr_window_timestamps)

        # Handle the remainder (last window)
        if data_max % args.window_size != 0:
            last_window_start = int(data_max) - (int(data_max) % int(args.window_size))
            features, previous_count, previous_density, total_density_sum, total_count_sum, total_windows, curr_window_timestamps = self.process_window(
                start=last_window_start,
                end=data_max,
                timestamps=timestamps,
                previous_count=previous_count,
                previous_density=previous_density,
                total_density_sum=total_density_sum,
                total_count_sum=total_count_sum,
                total_windows=total_windows,
                rolling_densities=rolling_densities,
                rolling_counts=rolling_counts,
                args=args
            )
            edge_features.append(features)
            window_timestamps.append(curr_window_timestamps)

        print("Edge feaures length: ", len(edge_features), flush=True)

        edge_features = torch.stack(edge_features)

        return edge_features, window_timestamps
    
    def process_window(self, start, end, timestamps, previous_count, previous_density, total_density_sum, total_count_sum, total_windows, rolling_densities, rolling_counts, args):
        # print("Processing window from ", flush=True)
        curr_window_timestamps = timestamps[(timestamps >= start) & (timestamps < end)]
        
        # Get the number of timestamps in the current window
        count = np.sum((timestamps >= start) & (timestamps < end))
        count_difference = count - previous_count
        previous_count = count

        # print("Processing window from ", start, " to ", end, flush=True)
        # Normalize the window for KDE
        grid_start = self.scaler.transform(np.array([[start]]))[0, 0]
        grid_end = self.scaler.transform(np.array([[end]]))[0, 0]
        grid = np.linspace(grid_start, grid_end, 6)

        # print("Grid start: ", grid_start, " Grid end: ", grid_end, flush=True)

        kde_density = np.exp(self.enhanced_edge.kde.score_samples(grid.reshape(-1, 1)))
        avg_density = np.mean(kde_density)  # Average density for the current window
        max_density = np.max(kde_density)  # Maximum density for the current window

        density_difference = max_density - previous_density

        # Update the total density sum and window count
        total_count_sum += count
        total_density_sum += avg_density
        total_windows += 1

        overall_avg_density = total_density_sum / total_windows
        overall_avg_count = total_count_sum / total_windows

        # rolling_densities.append(avg_density)
        # rolling_counts.append(count)

        # if len(rolling_densities) >= 3:
        #     rolling_density = np.mean(rolling_densities[-3:])
        # else:
        #     rolling_density = np.mean(rolling_densities)

        # if len(rolling_counts) >= 3:
        #     rolling_count = np.mean(rolling_counts[-3:])
        # else:
        #     rolling_count = np.mean(rolling_counts)

        if args.use_density:
            # print("Using density features", flush=True)
            features = [count, count_difference, overall_avg_count, density_difference, overall_avg_density, avg_density, max_density]
            # features = [overall_avg_density, avg_density, max_density]
        else:
            # features = [count, count_difference, overall_avg_count]
            features = [count, count_difference, overall_avg_count]
        # print("Count: ", count, " Count Difference: ", count_difference, " Overall Avg Count: ", overall_avg_count, " Overall Avg Density: ", overall_avg_density, " Avg Density: ", avg_density, " Max Density: ", max_density, flush=True)
        # features = [count, count_difference, overall_avg_count, overall_avg_density, avg_density, max_density]

        return torch.tensor(features, dtype=torch.float32), previous_count, previous_density, total_density_sum, total_count_sum, total_windows, curr_window_timestamps
    
    def partition_datasets(self, features):

        # Ensure timestamps are a NumPy array
        # timestamps = np.asarray(edge.timestamps)

        # Randomly select 20% of the timestamps for held-out data
        num_held_out = int(len(features) * 0.2)
        held_out_indices = np.random.choice(len(features), num_held_out, replace=False)

        # Get the remaining indices (not in held-out data)
        mask = np.ones(len(features), dtype=bool)
        mask[held_out_indices] = False
        remaining_indices = np.where(mask)[0]

        # Split the remaining indices into older and recent halves
        midpoint = len(remaining_indices) // 2
        older_indices = remaining_indices[:midpoint]
        recent_indices = remaining_indices[midpoint:]

        np.random.shuffle(older_indices)
        np.random.shuffle(recent_indices)

        assert len(set(held_out_indices).intersection(set(older_indices))) == 0
        assert len(set(held_out_indices).intersection(set(recent_indices))) == 0
        assert len(set(older_indices).intersection(set(recent_indices))) == 0

        self.held_out_features = features[held_out_indices]

        return features[older_indices], features[recent_indices]
    

    def generate_pairs(self, older_data, recent_data):
        pairs = []
        labels = []
        
        # Use the smaller dataset size to determine the number of pairs
        num_pairs = min(len(older_data), len(recent_data))
        
        for i in range(num_pairs):
            sample_old = older_data[i]
            sample_recent = recent_data[i]
            
            pair, label = self.generate_pair(sample_old, sample_recent)
            pairs.append(pair)
            labels.append(label)

        labels = torch.stack(labels)  # Shape: (num_pairs,)
        
        return pairs, labels
    
    def generate_pair(self, sample_old, sample_recent):
        if np.random.random() < 0.5:
            # Order: (old, recent) → label 1 means the second sample is more recent.
            pair = (sample_old, sample_recent)
            label = torch.tensor([1.0], dtype=torch.float32)
        else:
            # Order: (recent, old) → label 0 means the first sample is more recent.
            pair = (sample_recent, sample_old)
            label = torch.tensor([0.0], dtype=torch.float32)

        return pair, label

    def train_test_split(self, older_data, recent_data):
        # Remove the validation split
        older_train, older_test = np.split(older_data, [int(0.8 * len(older_data))])
        recent_train, recent_test = np.split(recent_data, [int(0.8 * len(recent_data))])

        input_dim = older_train.shape[1] + recent_train.shape[1]  # Concatenate the features of both samples
        
        return older_train, older_test, recent_train, recent_test, input_dim

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use Xavier initialization for weights
            nn.init.xavier_uniform_(module.weight)
            # Initialize biases to zero
            if module.bias is not None:
                nn.init.zeros_(module.bias)                

    def init_model(self, input_dim):
        # hidden_dim = 4
        # num_layers = 2
        # learning_rate = 0.001
        # num_epochs = 10
        # dropout = 0.10

        hidden_dim = 12
        num_layers = 2
        learning_rate = 0.001
        num_epochs = 1000
        dropout = 0.10

        self.model = RecencyClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.model.apply(self.initialize_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()  # Use CrossEntropyLoss for softmax outputs

        return num_epochs

    def train_model(self, num_epochs, training_pairs, training_labels, validation_pairs, validation_labels):
        print("Length of training pairs: ", len(training_pairs), flush=True)
        print(f"Number of unique pairs: {len(set(training_pairs))}", flush=True)
        print(f"Label distribution: {sum(training_labels)} positive, {len(training_labels) - sum(training_labels)} negative", flush=True)

        downsampled_train_pairs = []
        downsampled_train_labels = []

        # Separate positive and negative pairs
        positive_pairs = [(pair, label) for pair, label in zip(training_pairs, training_labels) if label == 1]
        negative_pairs = [(pair, label) for pair, label in zip(training_pairs, training_labels) if label == 0]

        # Determine the smaller group size
        min_size = min(len(positive_pairs), len(negative_pairs))

        for i in range(0, min_size):
            pair, label = positive_pairs[i]
            downsampled_train_pairs.append(pair)
            downsampled_train_labels.append(label)

        for i in range(0, min_size):
            pair, label = negative_pairs[i]
            downsampled_train_pairs.append(pair)
            downsampled_train_labels.append(label)

        # Shuffle the downsampled training pairs and labels
        combined = list(zip(downsampled_train_pairs, downsampled_train_labels))
        np.random.shuffle(combined)
        train_pairs, train_labels = zip(*combined)

        print("Length of training pairs after balancing: ", len(train_pairs), flush=True)
        print(f"Label distribution after balancing: {sum(train_labels)} positive, {len(train_labels) - sum(train_labels)} negative", flush=True)

        batch_size = 16
        self.model.train()

        best_val_loss = float('inf')  # Initialize the best validation loss
        epochs_without_improvement = 0  # Counter for early stopping
        patience = 5

        for epoch in range(num_epochs):
            # Reset training loss for the epoch
            self.total_train_loss = 0.0
            self.train_n = 0

            # Training loop
            for i in range(0, len(training_pairs), batch_size):
                batch_pairs = training_pairs[i:i + batch_size]
                batch_labels = training_labels[i:i + batch_size]

                batch_inputs = torch.stack([torch.cat(pair) for pair in batch_pairs])  # Shape: (batch_size, 2 * feature_dim)
                self.total_train_loss += self.train_point(batch_inputs, batch_labels)
                self.train_n += batch_size

            train_loss = self.total_train_loss / self.train_n
            print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss:.4f}", flush=True)

            # Validation loop
            val_loss, val_accuracy = self.evaluate_model(validation_pairs, validation_labels)
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}", flush=True)
            print("Best validation loss so far: ", best_val_loss, flush=True)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the best model
                # torch.save(self.model.state_dict(), "best_recency_classifier.pth")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.", flush=True)
                    break

        #     print(f"Epoch {epoch+1}/{num_epochs} - Loss: {self.total_train_loss/self.train_n:.4f}", flush=True)
        # _, train_accuracy = self.evaluate_model(training_pairs, training_labels)
        # print(f"Training Accuracy: {train_accuracy:.4f}", flush=True)

    def train_point(self, batch_inputs, batch_labels):
        # inputs = torch.cat([sample1, sample2])

        outputs = self.model(batch_inputs)

        loss = self.loss_fn(outputs, batch_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        return loss.item()


    def evaluate_model(self, pairs, labels):
        print(f"Label distribution evaluate model: {sum(labels)} positive, {len(labels) - sum(labels)} negative")

        self.model.eval()
        total_loss = 0.0
        correct = 0
        loss_fn = nn.BCEWithLogitsLoss()  # Use CrossEntropyLoss for softmax outputs
        with torch.no_grad():
            all_predictions = []
            outputs_holder = []
            for (sample1, sample2), label in zip(pairs, labels):
                inputs = torch.cat([sample1, sample2])  # Shape: (2, num_features)
                outputs = self.model(inputs)

                outputs_holder.append(outputs.item())
                # Compute loss
                loss = loss_fn(outputs, label)
                total_loss += loss.item()

                # Get the predicted class
                prediction = (outputs > 0.5).long()  # Threshold the probability at 0.5
                all_predictions.append(prediction.item())
                # Print out unique prediction count
            
                correct += (prediction == label).sum().item()

            # print("Outputs: ", outputs_holder, flush=True)

        all_predictions_tensor = torch.tensor(all_predictions)
        unique_values, counts = torch.unique(all_predictions_tensor, return_counts=True)
        
        avg_loss = total_loss / len(pairs)
        accuracy = correct / len(pairs)
        print("Evaluation accuracy: ", accuracy, flush=True)
        print("Evaluation loss: ", avg_loss, flush=True)
        print("Prediction counts:", dict(zip(unique_values.tolist(), counts.tolist())), flush=True)
        return avg_loss, accuracy
    
    def detect_distribution_shift(self, new_sample):
        self.model.eval()
        # Pair the new data point with a random held-out (historical) sample.
        held_sample = random.choice(self.held_out_features)
        # print("held sample: ", held_sample, flush=True)
        # print("new sample: ", new_sample, flush=True)

        # pair = torch.cat([held_sample, new_sample])
        pair, label = self.generate_pair(held_sample, new_sample)
        inputs = torch.cat(pair)  # Concatenate the two sampless
        with torch.no_grad():
            outputs = self.model(inputs)
            # Compute a "score": the network should give high probability that the new sample is more recent.
            probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
            
            # Use the probability of the second sample being more recent
            p = probabilities.item()  # Probability that the new sample is more recent
            print("Probability (p): ", p, flush=True)

        prediction = 1 if p > 0.5 else 0
        Y_k = 1 if prediction == label else 0
        
        return Y_k
    
    def exponential_martingale(self, S_n, n, t=1, p=0.5, q=0.5):
        """
        Compute the exponential martingale M_n in a numerically stable way.
        
        Args:
            S_n (float): The cumulative sum of predictions (1 or 0) up to step n.
            n (int): The number of predictions so far.
            t (float): The parameter controlling the growth of the martingale (default: 1).
            p (float): Bernoulli parameter for "success" (default: 0.5).
            q (float): Bernoulli parameter for "failure" (default: 0.5).
        
        Returns:
            float: The value of the martingale M_n.
        """
        # Compute log(M_n) to avoid overflow
        log_numerator = t * S_n
        log_denominator = n * math.log(q + p * math.exp(t))
        log_martingale = log_numerator - log_denominator

        # Convert back to the original scale
        return math.exp(log_martingale)
    
    def update_model(self, prev_features, new_timestamps, length_old_windows, args):
        flattened_timestamps = [item for sublist in new_timestamps for item in sublist]

        # Convert to NumPy array and reshape
        new_timestamps_array = np.array(flattened_timestamps).reshape(-1, 1)
        print("New timestamps array shape: ", new_timestamps_array.shape, flush=True)       
        new_timestamps_scaled = self.scaler.transform(new_timestamps_array)
        print("New timestamps scaled shape: ", new_timestamps_scaled.shape, flush=True)
        
        new_timestamps_scaled_weights = np.ones_like(new_timestamps_scaled).squeeze()
        prev_points, prev_weights = self.enhanced_edge.gh3_pseudo_points()
        # prev_weights = prev_weights.reshape(-1, 1)  # Ensure prev_weights is a 2D array
        # new_timestamps_scaled_weights = new_timestamps_scaled_weights.reshape(-1, 1)  # Ensure new_timestamps_scaled_weights is a 2D array
        prev_points = prev_points.reshape(-1, 1)  # Ensure prev_points is a 2D array
        print("Previous points shape: ", prev_points.shape, flush=True)
        print("Previous weights shape: ", prev_weights.shape, flush=True)
        print("New timestamps scaled weights shape: ", new_timestamps_scaled_weights.shape, flush=True)
        
        all_timestamps = np.concatenate((prev_points, new_timestamps_scaled), axis=0)
        print("all timestamps shape: ", all_timestamps.shape, flush=True)
        all_weights = np.concatenate((prev_weights, new_timestamps_scaled_weights), axis=0)
        print("all weights shape: ", all_weights.shape, flush=True)

        self.enhanced_edge.fit(timestamps=all_timestamps, data_weights=all_weights, args=args)

        # Recreate the edge features because the kde curve of the edge has changed
        new_features, _ = self.create_edge_features(new_timestamps_array, args=args)
        print("Length of new features: ", len(new_features), flush=True)
        print("Prev features shape: ", prev_features.shape, flush=True)
        prev_features = prev_features[1:, :]  # Remove the first row        print("Length of prev features: ", len(prev_features), flush=True)
        # # Calling this will repopulate the held out features, new random points selected
        features = torch.cat([prev_features, new_features], dim=0)  # Concatenate the old and new features
        old_features, recent_features = self.partition_datasets(features)

        old_sample = random.choice(old_features)
        pair, label = self.generate_pair(old_sample, new_features[-1])

        sample_1, sample_2 = pair
        batch_inputs = torch.cat([sample_1, sample_2]).unsqueeze(0)  # Shape: (1, 2 * feature_dim)
        # batch_labels = torch.tensor([label], dtype=torch.float32).unsqueeze(1)  # Shape: (1, 1)
        batch_labels = label.unsqueeze(0)  # Shape: (1, 1)
        # print("Batch inputs shape: ", batch_inputs.shape, flush=True)
        # print("Batch labels shape: ", label.shape, flush=True)
        self.model.train()
        self.total_train_loss += self.train_point(batch_inputs, batch_labels)
        self.train_n += 1

        print("Online update average loss: ", self.total_train_loss / self.train_n, flush=True)
        return prev_features, new_features[-1]  # Return the updated previous features for the next iteration

class RecencyClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, num_layers=2, dropout=0.5):
        super(RecencyClassifier, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  # Add dropout after each hidden layer
            # layers.append(nn.Dropout(0.25))  # Add dropout after each hidden layer
        
        layers.append(nn.Linear(hidden_dim, 1))
        # layers.append(nn.Softmax(dim=1))  # Apply sigmoid to output probabilities
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
