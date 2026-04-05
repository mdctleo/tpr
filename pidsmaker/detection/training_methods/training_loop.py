"""Training loop for PIDS models.

Handles model training with:
- Self-supervised pretraining with multiple objectives
- Optional few-shot fine-tuning for attack detection
- Gradient accumulation for large graphs
- Early stopping with patience
- Memory tracking (GPU and CPU)
- Validation-based model selection
- Batch timing instrumentation for KDE analysis
"""

import copy
import os
import tracemalloc
from time import perf_counter as timer

import numpy as np
import torch
import wandb

from pidsmaker.factory import (
    build_model,
    optimizer_factory,
    optimizer_few_shot_factory,
)
from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils.utils import get_device, log, log_start, log_tqdm, set_seed

# Import KDE debug logging (optional - won't fail if not available)
try:
    from pidsmaker.kde_patch import log_kde_debug_stats, reset_kde_state, finalize_kde_training
    KDE_DEBUG_AVAILABLE = True
except ImportError:
    KDE_DEBUG_AVAILABLE = False

# Import batch timing instrumentation (optional)
try:
    from pidsmaker.utils.batch_timing import (
        init_global_tracker,
        get_global_tracker,
        is_tracking_enabled,
        BatchTimingTracker,
    )
    BATCH_TIMING_AVAILABLE = True
except ImportError:
    BATCH_TIMING_AVAILABLE = False

from . import inference_loop


def main(cfg):
    """Main training loop executing self-supervised pretraining and optional few-shot fine-tuning.

    Training process:
    1. Self-supervised pretraining on reconstruction/prediction objectives
    2. Optional few-shot fine-tuning on labeled attack data
    3. Validation-based model selection (best epoch or each epoch)
    4. Early stopping with configurable patience

    Args:
        cfg: Configuration with training hyperparameters (epochs, lr, patience, etc.)

    Returns:
        float: Best validation score achieved during training
    """
    set_seed(cfg)

    log_start(__file__)
    device = get_device(cfg)
    use_cuda = device == torch.device("cuda")

    # Reset the peak memory usage counter
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device=device)
    tracemalloc.start()

    train_data, val_data, test_data, max_node_num = get_preprocessed_graphs(cfg)

    model = build_model(
        data_sample=train_data[0][0], device=device, cfg=cfg, max_node_num=max_node_num
    )
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))

    # Initialize batch timing tracker if enabled
    # Works for both KDE configs (kde_params) and baseline configs (batch_timing_vectors_dir)
    batch_tracker = None
    enable_batch_timing = getattr(cfg.training, 'enable_batch_timing', False)
    
    if BATCH_TIMING_AVAILABLE and enable_batch_timing:
        try:
            # Determine KDE vectors directory and min_occurrences
            # Priority: kde_params.kde_vectors_dir (if set) > batch_timing_vectors_dir (if baseline config)
            kde_vectors_dir = None
            min_occurrences = 10  # default
            config_type = "unknown"
            
            # Check if kde_params has actual values (not just None placeholders)
            kde_params_kde_vectors_dir = getattr(cfg.kde_params, 'kde_vectors_dir', None) if hasattr(cfg, 'kde_params') else None
            
            if kde_params_kde_vectors_dir:
                # KDE config with kde_vectors_dir set
                kde_vectors_dir = kde_params_kde_vectors_dir
                min_occurrences = getattr(cfg.kde_params, 'min_occurrences', 10) or 10
                config_type = "KDE"
            elif hasattr(cfg.training, 'batch_timing_vectors_dir') and cfg.training.batch_timing_vectors_dir:
                # Baseline config - use batch_timing_vectors_dir for taint tracking only
                kde_vectors_dir = cfg.training.batch_timing_vectors_dir
                min_occurrences = getattr(cfg.training, 'batch_timing_min_occurrences', 10) or 10
                config_type = "baseline"
            
            if kde_vectors_dir:
                # Store batch timing results under evaluation task path
                timing_output_dir = os.path.join(cfg.evaluation._task_path, "batch_timing")
                
                # Get dataset-specific dimensions for msg tensor edge type extraction
                # CIC-IDS: (1, 3), DARPA E3: (3, 10), etc.
                _node_type_dim = getattr(cfg.dataset, 'num_node_types', 8)
                _edge_type_dim = getattr(cfg.dataset, 'num_edge_types', 16)
                
                batch_tracker = init_global_tracker(
                    dataset_name=cfg.dataset.name,
                    kde_vectors_dir=kde_vectors_dir,
                    output_dir=timing_output_dir,
                    device=device,
                    min_occurrences=min_occurrences,
                    node_type_dim=_node_type_dim,
                    edge_type_dim=_edge_type_dim,
                )
                log(f"Batch timing tracker initialized for {cfg.dataset.name} ({config_type} config)")
                log(f"Using KDE vectors from: {kde_vectors_dir} with min_occurrences={min_occurrences}")
                log(f"Dataset dimensions: node_type_dim={_node_type_dim}, edge_type_dim={_edge_type_dim}")
                log(f"Batch timing results will be saved to: {timing_output_dir}")
            else:
                log(f"Batch timing enabled but no kde_vectors_dir configured (need kde_params.kde_vectors_dir or batch_timing_vectors_dir)")
        except Exception as e:
            log(f"Failed to initialize batch timing tracker: {e}")
            batch_tracker = None

    run_evaluation = cfg.training_loop.run_evaluation
    assert run_evaluation in ["best_epoch", "each_epoch"], (
        f"Invalid run evaluation {run_evaluation}"
    )
    best_epoch_mode = run_evaluation == "best_epoch"

    num_epochs = cfg.training.num_epochs
    tot_loss = 0.0
    epoch_times = []
    peak_train_cpu_mem = 0
    peak_train_gpu_mem = 0
    test_stats = None
    patience = cfg.training.patience
    patience_counter = 0
    all_test_stats = []
    global_best_val_score = float("-inf")
    use_few_shot = cfg.training.decoder.use_few_shot
    grad_acc = cfg.training.grad_accumulation

    if use_few_shot:
        num_epochs += 1  # in few-shot, the first epoch is without ssl training

    for epoch in range(0, num_epochs):
        best_val_score, best_model, best_epoch = float("-inf"), None, None

        if not use_few_shot or (use_few_shot and epoch > 0):
            start = timer()
            tracemalloc.start()

            # Before each epoch, we reset the memory
            model.reset_state()
            model.to_fine_tuning(False)
            
            # Set current epoch for KDE patch to detect last epoch
            for name, module in model.named_modules():
                if hasattr(module, 'forward'):
                    module._current_epoch = epoch
                    module._max_epochs = num_epochs
            
            # Reset KDE state ONLY for epoch 0 (to clear any previous run's data)
            # For epochs 1+, preserve KDE vectors computed after epoch 0
            if KDE_DEBUG_AVAILABLE and epoch == 0:
                try:
                    reset_kde_state(model)
                    log("Reset KDE state for fresh start in epoch 0")
                except Exception as e:
                    log(f"KDE state reset failed: {e}")

            loss_acc = torch.zeros(1, device=device)
            tot_loss = 0
            batch_idx = 0
            for dataset in train_data:
                for i, g in enumerate(log_tqdm(dataset, "Training")):
                    g.to(device=device)
                    g = remove_attacks_if_needed(g, cfg)
                    model.train()
                    optimizer.zero_grad()

                    # Time the forward pass if tracker is available
                    if batch_tracker is not None:
                        batch_tracker.set_epoch(epoch)
                        batch_tracker.set_split('train')
                        
                        # Start backward timing
                        backward_start = None
                        
                        def forward_fn(batch):
                            return model(batch)
                        
                        results, forward_time = batch_tracker.time_forward(g, forward_fn, phase='train')
                        loss = results["loss"]
                        loss_acc += loss
                        tot_loss += loss.item()

                        if (i + 1) % grad_acc == 0:
                            backward_start = timer()
                            loss_acc.backward()
                            optimizer.step()
                            backward_time_ms = (timer() - backward_start) * 1000
                            batch_tracker.record_backward_time(backward_time_ms)
                            loss_acc = torch.zeros(1, device=device)
                        
                        # Log tainted batches periodically
                        if batch_idx % 100 == 0 and batch_tracker.results:
                            last_result = batch_tracker.results[-1]
                            if last_result.kde_eligible_edges > 0:
                                batch_tracker.log_batch_detail(last_result)
                    else:
                        results = model(g)
                        loss = results["loss"]
                        loss_acc += loss
                        tot_loss += loss.item()

                        if (i + 1) % grad_acc == 0:
                            loss_acc.backward()
                            optimizer.step()
                            loss_acc = torch.zeros(1, device=device)

                    batch_idx += 1
                    g.to("cpu")
                    if use_cuda:
                        torch.cuda.empty_cache()

                # Last batch
                if loss_acc > 0:
                    loss_acc.backward()
                    optimizer.step()

            tot_loss /= sum(len(dataset) for dataset in train_data)
            epoch_times.append(timer() - start)

            _, peak_inference_cpu_memory = tracemalloc.get_traced_memory()
            peak_train_cpu_mem = max(peak_train_cpu_mem, peak_inference_cpu_memory / (1024**3))
            tracemalloc.stop()

            if use_cuda:
                peak_inference_gpu_memory = torch.cuda.max_memory_allocated(device=device) / (
                    1024**3
                )
                peak_train_gpu_mem = max(peak_train_gpu_mem, peak_inference_gpu_memory)
                torch.cuda.reset_peak_memory_stats(device=device)

            log(
                f"[@epoch{epoch:02d}] Training finished - GPU memory: {peak_train_gpu_mem:.2f} GB | CPU memory: {peak_train_cpu_mem:.2f} GB | Mean Loss: {tot_loss:.4f}",
                return_line=True,
            )
            
            # KDE Debug logging after training
            if KDE_DEBUG_AVAILABLE:
                try:
                    log_kde_debug_stats(model, epoch, "training")
                except Exception as e:
                    log(f"KDE debug logging failed: {e}")
                
                # Build KDE vectors after FIRST epoch (epoch 0) for use in all subsequent epochs
                if epoch == 0:
                    try:
                        log("Building KDE vectors after first epoch...")
                        finalize_kde_training(model)
                        log("KDE vectors built successfully - will be used for all subsequent epochs")
                    except Exception as e:
                        log(f"KDE finalization failed: {e}")

        # Few-shot learning fine tuning
        if use_few_shot:
            model.to_fine_tuning(True)
            optimizer = optimizer_few_shot_factory(cfg, parameters=set(model.parameters()))

            num_epochs_few_shot = cfg.training.decoder.few_shot.num_epochs_few_shot
            patience_few_shot = cfg.training.decoder.few_shot.patience_few_shot

            for tuning_epoch in range(0, num_epochs_few_shot):
                model.reset_state()

                loss_acc = torch.zeros(1, device=device)
                tot_loss = 0
                for dataset in train_data:
                    for g in log_tqdm(dataset, "Fine-tuning"):
                        if 1 in g.y:
                            g.to(device=device)
                            model.train()
                            optimizer.zero_grad()

                            results = model(g)
                            loss = results["loss"]
                            loss_acc += loss
                            tot_loss += loss.item()

                            if (i + 1) % grad_acc == 0:
                                loss_acc.backward()
                                optimizer.step()
                                loss_acc = torch.zeros(1, device=device)

                            g.to("cpu")
                            if use_cuda:
                                torch.cuda.empty_cache()

                    # Last batch
                    if loss_acc > 0:
                        loss_acc.backward()
                        optimizer.step()

                tot_loss /= sum(len(dataset) for dataset in train_data)

                # Validation
                val_stats = inference_loop.main(
                    cfg=cfg,
                    model=model,
                    val_data=val_data,
                    test_data=test_data,
                    epoch=epoch,
                    split="val",
                    logging=False,
                )
                val_loss = val_stats["val_loss"]
                val_score = val_stats["val_score"]

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                    patience_counter = 0
                else:
                    patience_counter += 1

                if val_score > global_best_val_score:
                    global_best_val_score = val_score
                    best_epoch = epoch

                log(
                    f"[@epoch{tuning_epoch:02d}] Fine-tuning - Train Loss: {tot_loss:.5f} | Val Loss: {val_loss:.4f}",
                    return_line=True,
                )

                if patience_counter >= patience_few_shot:
                    log(f"Early stopping: best few-shot loss is {best_val_score:.4f}")
                    break

            model.load_state_dict(best_model)
            model.to_device(device)

        # model_path = os.path.join(gnn_models_dir, f"model_epoch_{epoch}")
        # save_model(model, model_path, cfg)

        # Test
        if (epoch + 1) % 2 == 0 or epoch == 0:
            test_stats = inference_loop.main(
                cfg=cfg,
                model=model,
                val_data=val_data,
                test_data=test_data,
                epoch=epoch,
                split="all",
            )
            all_test_stats.append(test_stats)

            wandb.log(
                {
                    "epoch": epoch,
                    "train_epoch": epoch,
                    "train_loss": round(tot_loss, 4),
                    "val_score": round(test_stats["val_score"], 4),
                    "val_loss": round(test_stats["val_loss"], 4),
                    "test_loss": round(test_stats["test_loss"], 4),
                }
            )

    # After training
    if best_epoch_mode:
        model.load_state_dict(best_model)
        test_stats = inference_loop.main(
            cfg=cfg,
            model=model,
            val_data=val_data,
            test_data=test_data,
            epoch=best_epoch,
            split="test",
        )

    wandb.log(
        {
            "best_epoch": best_epoch,
            "train_epoch_time": round(np.mean(epoch_times), 2),
            "val_score": round(best_val_score, 5),
            "peak_train_cpu_memory": round(peak_train_cpu_mem, 3),
            "peak_train_gpu_memory": round(peak_train_gpu_mem, 3),
            "peak_inference_cpu_memory": round(
                np.max([d["peak_inference_cpu_memory"] for d in all_test_stats]), 3
            ),
            "peak_inference_gpu_memory": round(
                np.max([d["peak_inference_gpu_memory"] for d in all_test_stats]), 3
            ),
            "time_per_batch_inference": round(
                np.mean([d["time_per_batch_inference"] for d in all_test_stats]), 3
            ),
        }
    )

    # Log batch timing summary and save results
    if batch_tracker is not None:
        try:
            log("\n" + "=" * 60)
            log("BATCH TIMING ANALYSIS")
            log("=" * 60)
            batch_tracker.log_summary()
            
            # Save detailed results
            results_file = batch_tracker.save_results(f"batch_timing_{cfg.dataset.name}.json")
            tainted_file = batch_tracker.save_detailed_tainted_report(f"tainted_batches_{cfg.dataset.name}.json")
            
            # Log to wandb
            tainted_batches = batch_tracker.get_tainted_batches(min_taint_ratio=0.0)
            tainted_count = len([r for r in tainted_batches if r.kde_eligible_edges > 0])
            total_batches = len(batch_tracker.results)
            
            wandb.log({
                "batch_timing/total_batches": total_batches,
                "batch_timing/tainted_batches": tainted_count,
                "batch_timing/taint_percentage": tainted_count / total_batches * 100 if total_batches > 0 else 0,
            })
            
            log(f"Batch timing results saved to {results_file}")
            log(f"Tainted batches report saved to {tainted_file}")
        except Exception as e:
            log(f"Failed to save batch timing results: {e}")

    return best_val_score


def remove_attacks_if_needed(graph, cfg):
    """Remove attack edges from graph for self-supervised training if configured.

    Args:
        graph: Graph batch with labels in graph.y
        cfg: Configuration with few_shot.include_attacks_in_ssl_training setting

    Returns:
        graph: Original graph or filtered graph without attacks (y=1)
    """
    if not cfg.training.decoder.few_shot.include_attacks_in_ssl_training:
        if 1 in graph.y:
            return graph.clone()[graph.y != 1]
    return graph
