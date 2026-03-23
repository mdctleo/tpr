"""Training loop for PIDS models.

Handles model training with:
- Self-supervised pretraining with multiple objectives
- Optional few-shot fine-tuning for attack detection
- Gradient accumulation for large graphs
- Early stopping with patience
- Memory tracking (GPU and CPU)
- Validation-based model selection
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
    from pidsmaker.utils.batch_timing import init_global_tracker, get_global_tracker
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

    # Initialize batch timing tracker for KDE-tainted batch analysis
    tracker = None
    if BATCH_TIMING_AVAILABLE:
        kde_params = getattr(cfg, 'kde_params', None)
        if kde_params is not None:
            try:
                kp = dict(kde_params) if not isinstance(kde_params, dict) else kde_params
                tracker = init_global_tracker(
                    dataset_name=cfg.dataset.name,
                    kde_vectors_dir=kp.get('kde_vectors_dir', 'kde_vectors'),
                    output_dir="timing_results",
                    device=device,
                    min_occurrences=kp.get('min_occurrences', 10),
                )
                log(f"Batch timing tracker initialized with {len(tracker.kde_eligible_edges)} KDE-eligible edges")
            except Exception as e:
                log(f"Failed to initialize batch timing tracker: {e}")
                tracker = None

    model = build_model(
        data_sample=train_data[0][0], device=device, cfg=cfg, max_node_num=max_node_num
    )
    optimizer = optimizer_factory(cfg, parameters=set(model.parameters()))

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
            for dataset in train_data:
                for i, g in enumerate(log_tqdm(dataset, "Training")):
                    g.to(device=device)
                    g = remove_attacks_if_needed(g, cfg)
                    model.train()
                    optimizer.zero_grad()

                    if tracker:
                        tracker.start_batch()

                    results = model(g)
                    loss = results["loss"]
                    loss_acc += loss
                    tot_loss += loss.item()

                    if (i + 1) % grad_acc == 0:
                        loss_acc.backward()
                        optimizer.step()
                        loss_acc = torch.zeros(1, device=device)

                    if tracker:
                        tracker.end_batch_auto(
                            batch_idx=i, epoch=epoch, batch=g, phase='train',
                        )

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

    # Save batch timing results and log summary
    batch_timing_metrics = {}
    if tracker:
        try:
            tracker.log_summary()
            tracker.save_results("batch_timing_results.json")
            tracker.save_detailed_tainted_report("tainted_batches_report.json")
            batch_timing_metrics = tracker.get_wandb_metrics("batch_timing", phase_filter="train")
            log(f"Batch timing results saved ({len(tracker.results)} batches tracked)")
        except Exception as e:
            log(f"Batch timing save failed: {e}")

    final_metrics = {
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
    final_metrics.update(batch_timing_metrics)
    wandb.log(final_metrics)

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
