import os
import random
import time
import tracemalloc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from pidsmaker.utils.utils import (
    calculate_average_from_file,
    get_device,
    log,
    log_tqdm,
    ns_time_to_datetime_US,
    set_seed,
)

# Import KDE debug logging (optional - won't fail if not available)
try:
    from pidsmaker.kde_patch import log_kde_debug_stats
    KDE_DEBUG_AVAILABLE = True
except ImportError:
    KDE_DEBUG_AVAILABLE = False

# Import batch timing (optional - won't fail if not available)
try:
    from pidsmaker.utils.batch_timing import BatchTimingTracker, BatchTimingResult, load_kde_eligible_edges
    BATCH_TIMING_AVAILABLE = True
except ImportError:
    BATCH_TIMING_AVAILABLE = False


@torch.no_grad()
def test_edge_level(
    data,
    model,
    split,
    model_epoch_file,
    cfg,
    device,
):
    model.eval()

    start_time = data.t[0]
    all_losses = []

    validation = split == "val"

    results = model(data, inference=True, validation=validation)
    each_edge_loss = results["loss"]
    all_losses.extend(each_edge_loss.cpu().numpy().tolist())

    # If the data has been reindexed in the loader or batched, we retrieve original node IDs
    # to later find the labels
    edge_index = data.original_edge_index

    srcnodes = edge_index[0, :].cpu().numpy()
    dstnodes = edge_index[1, :].cpu().numpy()
    t_vars = data.t.cpu().numpy()
    losses = each_edge_loss.cpu().numpy()
    edge_types = (data.edge_type.max(dim=1).indices + 1).cpu().numpy()

    # if 1 in data.y:
    #     log(f"Mean score of fake malicious edges: {losses[data.y.cpu() == 1].mean():.4f}")
    #     log(f"Mean score of benign malicious edges: {losses[data.y.cpu() == 0].mean():.4f}")

    edge_df = pd.DataFrame(
        {
            "loss": losses.astype(float),
            "srcnode": srcnodes.astype(int),
            "dstnode": dstnodes.astype(int),
            "time": t_vars.astype(int),
            "edge_type": edge_types.astype(int),
        }
    )

    # Here is a checkpoint, which records all edge losses in the current time window
    time_interval = (
        ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(edge_df["time"].max())
    )

    logs_dir = os.path.join(cfg.training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    csv_file = os.path.join(logs_dir, time_interval + ".csv")

    edge_df.to_csv(csv_file, sep=",", header=True, index=False, encoding="utf-8")
    return all_losses


@torch.no_grad()
def test_node_level(
    data,
    model,
    split,
    model_epoch_file,
    cfg,
    device,
):
    model.eval()

    node_list = []
    start_time = data.t[0]
    end_time = data.t[-1]
    losses = []

    validation = split == "val"

    data = data.to(device)

    results = model(data, inference=True, validation=validation)
    loss = results["loss"]
    losses.extend(loss.cpu().numpy().tolist())
    n_id = getattr(data, "original_n_id_tgn", getattr(data, "original_n_id"))

    # ThreaTrace code
    if cfg.evaluation.node_evaluation.threshold_method == "threatrace":
        out = results["out"]
        pred = out.max(1)[1]
        pro = F.softmax(out, dim=1)
        pro1 = pro.max(1)
        for i in range(len(out)):
            pro[i][pro1[1][i]] = -1
        pro2 = pro.max(1)

        node_type_num = data.node_type.argmax(1)
        for i in range(len(out)):
            if pro2[0][i] != 0:
                score = pro1[0][i] / pro2[0][i]
            else:
                score = pro1[0][i] / 1e-5
            score = torch.log(score + 1e-12)  # we do that or the score is much too high
            score = max(score.item(), 0)

            node = n_id[i].item()
            correct_pred = int((node_type_num[i] == pred[i]).item())

            temp_dic = {
                "node": node,
                "loss": float(loss[i].item()),
                "threatrace_score": score,
                "correct_pred": correct_pred,
            }
            node_list.append(temp_dic)

    # Flash code
    elif cfg.evaluation.node_evaluation.threshold_method == "flash":
        out = results["out"]
        pred = out.max(1)[1]
        sorted, indices = out.sort(dim=1, descending=True)
        eps = 1e-6
        conf = (sorted[:, 0] - sorted[:, 1]) / (sorted[:, 0] + eps)
        conf = (conf - conf.min()) / conf.max() if conf.max() > 0 else conf

        node_type_num = data.node_type.argmax(1)
        for i in range(len(out)):
            score = max(conf[i].item(), 0)

            node = n_id[i].item()
            correct_pred = int((node_type_num[i] == pred[i]).item())

            temp_dic = {
                "node": node,
                "loss": float(loss[i].item()),
                "flash_score": score,
                "correct_pred": correct_pred,
            }
            node_list.append(temp_dic)

    # Magic codes
    elif cfg.evaluation.node_evaluation.threshold_method == "magic":
        os.makedirs(cfg.training._magic_dir, exist_ok=True)
        if split == "val":
            x_train, _, _ = model.embed(data, inference=True)
            x_train = x_train.cpu().numpy()
            num_nodes = x_train.shape[0]
            sample_size = 5000 if num_nodes > 5000 else num_nodes
            sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
            x_train_sampled = x_train[sample_indices]
            x_train_mean = x_train_sampled.mean(axis=0)
            x_train_std = x_train_sampled.std(axis=0)
            x_train_sampled = (x_train_sampled - x_train_mean) / x_train_std

            x_train_sampled = pd.DataFrame.from_records(x_train_sampled)

            n_neighbors = 10
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train_sampled)
            idx = list(range(x_train_sampled.shape[0]))
            random.shuffle(idx)
            try:
                sample = x_train_sampled.iloc[idx[: min(50000, x_train_sampled.shape[0])]]
                distances_train, _ = nbrs.kneighbors(
                    sample, n_neighbors=min(len(sample), n_neighbors)
                )
            except KeyError as e:
                log(f"KeyError encountered: {e}")
                log(f"Available columns in x_train: {x_train_sampled.columns}")
                raise
            mean_distance_train = distances_train.mean().mean()
            if mean_distance_train == 0:
                mean_distance_train = 1e-9

            train_distance_file = os.path.join(cfg.training._magic_dir, "train_distance.txt")
            with open(train_distance_file, "a") as f:
                f.write(f"{mean_distance_train}\n")

            for i, node in enumerate(n_id):
                temp_dic = {
                    "node": node.item(),
                    "loss": float(loss[i].item()),
                }
                node_list.append(temp_dic)

        elif split == "test":
            train_distance_file = os.path.join(cfg.training._magic_dir, "train_distance.txt")
            mean_distance_train = calculate_average_from_file(train_distance_file)

            x_test, _, _ = model.embed(data, inference=True)
            x_test = x_test.cpu().numpy()
            num_nodes = x_test.shape[0]
            sample_size = 5000 if num_nodes > 5000 else num_nodes
            sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
            x_test_sampled = x_test[sample_indices]
            x_test_mean = x_test_sampled.mean(axis=0)
            x_test_std = x_test_sampled.std(axis=0)
            x_test_sampled = (x_test_sampled - x_test_mean) / x_test_std

            torch.cuda.empty_cache()
            x_test_sampled = pd.DataFrame.from_records(x_test_sampled)

            n_neighbors = 10
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_test_sampled)

            distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
            distances = distances.mean(axis=1)
            # distances = distances.to_numpy()
            score = distances / mean_distance_train
            score = score.tolist()

            for i, node in enumerate(n_id):
                temp_dic = {
                    "node": node.item(),
                    "magic_score": float(score[i]),
                    "loss": float(loss[i].item()),
                }
                node_list.append(temp_dic)

    else:
        for i, node in enumerate(n_id):
            temp_dic = {
                "node": node.item(),
                "loss": float(loss[i].item()),
            }
            node_list.append(temp_dic)

    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(end_time)

    logs_dir = os.path.join(cfg.training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    csv_file = os.path.join(logs_dir, time_interval + ".csv")

    df = pd.DataFrame(node_list)
    df.to_csv(csv_file, sep=",", header=True, index=False, encoding="utf-8")
    return losses


def main(cfg, model, val_data, test_data, epoch, split, logging=True):
    set_seed(cfg)

    if split == "all":
        splits = [(val_data, "val"), (test_data, "test")]
    elif split == "val":
        splits = [(val_data, "val")]
    elif split == "test":
        splits = [(test_data, "test")]
    else:
        raise ValueError(f"Invalid split {split}")

    inference_device = cfg.training.inference_device
    if inference_device is not None:
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid inference device {device}")
        device = torch.device(device)
    else:
        device = get_device(cfg)
    use_cuda = device == torch.device("cuda")
    model.to_device(device)

    model_epoch_file = f"model_epoch_{epoch}"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)

    # Initialize batch timing tracker for inference
    # Works for both KDE configs (kde_params) and baseline configs (batch_timing_vectors_dir)
    batch_tracker = None
    enable_batch_timing = getattr(cfg.training, 'enable_batch_timing', False)
    
    if BATCH_TIMING_AVAILABLE and enable_batch_timing:
        try:
            # Determine KDE vectors directory
            # Priority: kde_params.kde_vectors_dir (if set) > batch_timing_vectors_dir (if baseline config)
            kde_vectors_dir = None
            config_type = "unknown"
            
            # Check if kde_params has actual values (not just None placeholders)
            kde_params_kde_vectors_dir = getattr(cfg.kde_params, 'kde_vectors_dir', None) if hasattr(cfg, 'kde_params') else None
            
            if kde_params_kde_vectors_dir:
                # KDE config with kde_vectors_dir set
                kde_vectors_dir = kde_params_kde_vectors_dir
                config_type = "KDE"
            elif hasattr(cfg.training, 'batch_timing_vectors_dir') and cfg.training.batch_timing_vectors_dir:
                # Baseline config - use batch_timing_vectors_dir for taint tracking only
                kde_vectors_dir = cfg.training.batch_timing_vectors_dir
                config_type = "baseline"
            
            if kde_vectors_dir:
                # Load KDE edges from .pt file
                kde_vectors_file = os.path.join(kde_vectors_dir, f"{cfg.dataset.name}_kde_vectors.pt")
                
                if os.path.exists(kde_vectors_file):
                    kde_eligible_edges, edge_occurrence_counts = load_kde_eligible_edges(kde_vectors_file, device=device)
                else:
                    kde_eligible_edges = set()
                    edge_occurrence_counts = {}
                    log(f"KDE vectors file not found: {kde_vectors_file}")
                
                if kde_eligible_edges:
                    # Store batch timing results under evaluation task path
                    timing_output_dir = os.path.join(cfg.evaluation._task_path, "batch_timing")
                    
                    batch_tracker = BatchTimingTracker(
                        kde_eligible_edges=kde_eligible_edges,
                        edge_occurrence_counts=edge_occurrence_counts,
                        output_dir=timing_output_dir,
                        device=device,
                    )
                    log(f"Inference batch timing enabled ({config_type} config) with {len(kde_eligible_edges)} KDE-eligible edges")
                    log(f"Inference batch timing results will be saved to: {timing_output_dir}")
            else:
                log(f"Inference batch timing enabled but no kde_vectors_dir configured")
        except Exception as e:
            log(f"Failed to initialize inference batch timing tracker: {e}")
            batch_tracker = None

    val_score = 0.0
    peak_inference_cpu_mem = 0
    peak_inference_gpu_mem = 0
    tpb = []
    split2loss = {}

    for dataset, split_name in splits:
        desc = "Validation" if split_name == "val" else "Testing"

        tracemalloc.start()

        all_losses = []
        batch_idx = 0
        for graphs in dataset:
            for g in log_tqdm(graphs, desc=desc, logging=logging):
                g.to(device=device)

                # Start batch timing (for CUDA events)
                if batch_tracker is not None:
                    batch_tracker.start_batch()

                s = time.time()
                test_fn = test_node_level if cfg._is_node_level else test_edge_level
                losses = test_fn(
                    data=g,
                    model=model,
                    split=split_name,
                    model_epoch_file=model_epoch_file,
                    cfg=cfg,
                    device=device,
                )
                all_losses.extend(losses)
                tpb.append(time.time() - s)

                # End batch timing and record using GPU-optimized analyze_batch
                if batch_tracker is not None:
                    # Use GPU-accelerated analyze_batch() instead of CPU-based edge extraction
                    # This keeps edge type extraction and KDE lookup on GPU
                    analysis = batch_tracker.analyze_batch(g)
                    
                    # Calculate elapsed time using CUDA events
                    if batch_tracker._use_cuda:
                        batch_tracker._batch_cuda_end.record()
                        torch.cuda.synchronize()
                        forward_time_ms = batch_tracker._batch_cuda_start.elapsed_time(batch_tracker._batch_cuda_end)
                    else:
                        forward_time_ms = (time.perf_counter() - batch_tracker._batch_start_time) * 1000
                    
                    # Determine split from phase
                    phase = f"inference_{split_name}"
                    split = 'test'
                    if 'val' in phase:
                        split = 'val'
                    elif 'train' in phase:
                        split = 'train'
                    
                    # Create and store result (BatchTimingResult imported at top of file)
                    result = BatchTimingResult(
                        batch_id=batch_idx,
                        phase=phase,
                        epoch=-1,  # -1 indicates inference
                        split=split,
                        total_edges=analysis['total_edges'],
                        kde_eligible_edges=analysis['kde_eligible'],
                        non_kde_edges=analysis['non_kde'],
                        taint_ratio=analysis['taint_ratio'],
                        forward_time_ms=forward_time_ms,
                        total_time_ms=forward_time_ms,
                        edges_that_could_be_reduced=analysis['edges_could_reduce'],
                        total_timestamps=analysis['total_timestamps'],
                        kde_eligible_timestamps=analysis['kde_eligible_timestamps'],
                    )
                    
                    batch_tracker.results.append(result)
                    batch_tracker._update_summary(result)
                    batch_tracker.batch_counter += 1
                    batch_idx += 1

                g.to("cpu")  # Move graph back to CPU to free GPU memory for next batch
                if use_cuda:
                    torch.cuda.empty_cache()

        _, peak_inference_cpu_memory = tracemalloc.get_traced_memory()
        peak_inference_cpu_mem = max(peak_inference_cpu_mem, peak_inference_cpu_memory / (1024**3))
        tracemalloc.stop()

        if use_cuda:
            peak_inference_gpu_memory = torch.cuda.max_memory_allocated(device=device) / (1024**3)
            peak_inference_gpu_mem = max(peak_inference_gpu_mem, peak_inference_gpu_memory)
            torch.cuda.reset_peak_memory_stats(device=device)

        mean_loss = np.mean(all_losses)
        split2loss[split_name] = mean_loss

        if split_name == "val":
            val_score = model.get_val_ap()
            if logging:
                log(
                    f"[@epoch{epoch:02d}] Validation finished - Val Loss: {mean_loss:.4f}",
                    return_line=True,
                )
                # KDE Debug logging after validation
                if KDE_DEBUG_AVAILABLE:
                    try:
                        log_kde_debug_stats(model, epoch, "validation")
                    except Exception as e:
                        log(f"KDE debug logging failed: {e}")
        else:
            if logging:
                log(
                    f"[@epoch{epoch:02d}] Test finished - Test Loss: {mean_loss:.4f}",
                    return_line=True,
                )
                # KDE Debug logging after testing
                if KDE_DEBUG_AVAILABLE:
                    try:
                        log_kde_debug_stats(model, epoch, "testing")
                    except Exception as e:
                        log(f"KDE debug logging failed: {e}")

    # Log batch timing summary for inference
    if batch_tracker is not None:
        try:
            log("\n" + "=" * 60)
            log(f"INFERENCE BATCH TIMING ANALYSIS (epoch {epoch})")
            log("=" * 60)
            batch_tracker.log_summary()
            
            # Save results with inference prefix
            results_file = batch_tracker.save_results(f"inference_batch_timing_{cfg.dataset.name}_epoch{epoch}.json")
            tainted_file = batch_tracker.save_detailed_tainted_report(f"inference_tainted_batches_{cfg.dataset.name}_epoch{epoch}.json")
            
            log(f"Inference batch timing results saved to {results_file}")
            log(f"Inference tainted batches report saved to {tainted_file}")
        except Exception as e:
            log(f"Failed to save inference batch timing results: {e}")

    del model

    stats = {
        "val_score": val_score,
        "val_loss": split2loss.get("val", None),
        "test_loss": split2loss.get("test", None),
        "peak_inference_cpu_memory": peak_inference_cpu_mem,
        "peak_inference_gpu_memory": peak_inference_gpu_mem,
        "time_per_batch_inference": np.mean(tpb),
    }
    return stats
