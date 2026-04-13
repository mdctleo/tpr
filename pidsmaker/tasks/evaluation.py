import os

import wandb

from pidsmaker.detection.evaluation_methods import (
    edge_evaluation,
    node_evaluation,
    node_tw_evaluation,
    queue_evaluation,
    tw_evaluation,
)
from pidsmaker.detection.evaluation_methods.evaluation_utils import (
    build_attack_to_tw_indices,
    compute_tw_labels,
    listdir_sorted,
)
from pidsmaker.utils.labelling import get_GP_of_each_attack
from pidsmaker.utils.utils import log


def standard_evaluation(cfg, evaluation_fn):
    test_losses_dir = os.path.join(cfg.training._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.training._edge_losses_dir, "val")

    tw_to_malicious_nodes = compute_tw_labels(cfg)

    per_attack_enabled = getattr(cfg.dataset, "per_attack_test_graphs", False)
    test_file_to_attack_idx = dict(getattr(cfg.dataset, "test_file_to_attack_idx", []))
    attack_to_tw_indices = build_attack_to_tw_indices(cfg) if per_attack_enabled else {}
    attack_to_GPs_all = get_GP_of_each_attack(cfg) if per_attack_enabled else {}

    best_metrics = {
        "adp_score": float("-inf"),
        "discrimination": float("-inf"),
        "best_stats": None,
    }

    sorted_files = (
        listdir_sorted(test_losses_dir) if os.path.exists(test_losses_dir) else ["epoch_0"]
    )
    out_dir = cfg.evaluation._precision_recall_dir

    save_files_to_wandb = cfg._experiment != "uncertainty"

    for model_epoch_dir in sorted_files:
        log(f"[@{model_epoch_dir}] - Test Evaluation", pre_return_line=True)

        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        val_tw_path = os.path.join(val_losses_dir, model_epoch_dir)

        if per_attack_enabled and len(attack_to_tw_indices) > 0:
            per_attack_stats = []

            for test_file in cfg.dataset.test_files:
                attack_idx = test_file_to_attack_idx.get(test_file, None)
                if attack_idx is None:
                    continue

                selected_tw_indices = attack_to_tw_indices.get(attack_idx, [])
                if len(selected_tw_indices) == 0:
                    log(f"Skipping {test_file}: no TW indices found in edge embeds ordering")
                    continue

                attack_gp = attack_to_GPs_all.get(attack_idx, None)
                if attack_gp is None:
                    continue

                log(
                    f"[@{model_epoch_dir}] - Test Evaluation ({test_file}, TWs={len(selected_tw_indices)})"
                )
                attack_stats = evaluation_fn(
                    val_tw_path,
                    test_tw_path,
                    model_epoch_dir,
                    cfg,
                    tw_to_malicious_nodes=tw_to_malicious_nodes,
                    selected_tw_indices=selected_tw_indices,
                    ground_truth_nids=attack_gp["nids"],
                    attack_to_GPs={attack_idx: attack_gp},
                )
                attack_stats["attack_name"] = test_file
                attack_stats["attack_idx"] = attack_idx
                per_attack_stats.append((test_file, attack_stats))

                # Log per-attack confusion matrix summary
                log(
                    f"  [{test_file}] TP={attack_stats.get('tp', '?')}, "
                    f"FP={attack_stats.get('fp', '?')}, "
                    f"TN={attack_stats.get('tn', '?')}, "
                    f"FN={attack_stats.get('fn', '?')} | "
                    f"Prec={attack_stats.get('precision', '?')}, "
                    f"Rec={attack_stats.get('recall', '?')}, "
                    f"F1={attack_stats.get('fscore', '?')}"
                )

            # Aggregate per-attack stats to global epoch-level metrics.
            if len(per_attack_stats) == 0:
                stats = {
                    "adp_score": float("-inf"),
                    "discrimination": float("-inf"),
                }
            else:
                stats = {}
                scalar_keys = [
                    "precision",
                    "recall",
                    "fscore",
                    "ap",
                    "auc",
                    "accuracy",
                    "adp_score",
                    "discrimination",
                    "percent_detected_attacks",
                ]

                for key in scalar_keys:
                    vals = [s[key] for _, s in per_attack_stats if key in s]
                    if len(vals) > 0:
                        stats[key] = sum(vals) / len(vals)

                for test_file, s in per_attack_stats:
                    prefix = f"{test_file}/"
                    for key in [
                        "precision",
                        "recall",
                        "fscore",
                        "ap",
                        "auc",
                        "accuracy",
                        "adp_score",
                        "discrimination",
                        "tp",
                        "fp",
                        "tn",
                        "fn",
                    ]:
                        if key in s:
                            stats[prefix + key] = s[key]

                # Keep one valid image path for downstream wandb.save call.
                if "neat_scores_img_file" in per_attack_stats[0][1]:
                    stats["neat_scores_img_file"] = per_attack_stats[0][1]["neat_scores_img_file"]
                if "scores_file" in per_attack_stats[0][1]:
                    stats["scores_file"] = per_attack_stats[0][1]["scores_file"]
        else:
            stats = evaluation_fn(
                val_tw_path,
                test_tw_path,
                model_epoch_dir,
                cfg,
                tw_to_malicious_nodes=tw_to_malicious_nodes,
            )
        log(f"[@{model_epoch_dir}] - Stats")
        for k, v in stats.items():
            log(f"{k}: {v}")

        stats["epoch"] = int(model_epoch_dir.split("_")[-1])

        if save_files_to_wandb:
            # stats["simple_scores_img"] = wandb.Image(os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png"))

            scores = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
            if os.path.exists(scores):
                stats["scores_img"] = wandb.Image(scores)

            # pr = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
            # if os.path.exists(pr):
            #     stats["precision_recall_img"] = wandb.Image(pr)

            adp = os.path.join(out_dir, f"adp_curve_{model_epoch_dir}.png")
            if os.path.exists(adp):
                stats["adp_img"] = wandb.Image(adp)

            seen_scores = os.path.join(out_dir, f"seen_score_{model_epoch_dir}.png")
            if os.path.exists(seen_scores):
                stats["seen_scores_img"] = wandb.Image(seen_scores)

            discrim = os.path.join(out_dir, f"discrim_curve_{model_epoch_dir}.png")
            if os.path.exists(discrim):
                stats["discrim_img"] = wandb.Image(discrim)

        wandb.log(stats)

        best_metrics = best_metric_pick_best_epoch(stats, best_metrics, cfg)

    if save_files_to_wandb:
        # We only store the scores for the best run
        # wandb.save(best_metrics["stats"]["scores_file"], out_dir)
        wandb.save(best_metrics["stats"]["neat_scores_img_file"], out_dir)

    return best_metrics["stats"]


def best_metric_pick_best_epoch(stats, best_metrics, cfg):
    best_model_selection = cfg.evaluation.best_model_selection

    if best_model_selection == "best_adp":
        condition = (stats["adp_score"] > best_metrics["adp_score"]) or (
            stats["adp_score"] == best_metrics["adp_score"]
            and stats["discrimination"] > best_metrics["discrimination"]
        )

    elif best_model_selection == "best_discrimination":
        condition = stats["discrimination"] > best_metrics["discrimination"]

    else:
        raise ValueError(f"Invalid best model selection {best_model_selection}")

    if condition:
        best_metrics["adp_score"] = stats["adp_score"]
        best_metrics["discrimination"] = stats["discrimination"]
        best_metrics["stats"] = stats
    return best_metrics


def main(cfg):
    method = cfg.evaluation.used_method.strip()
    if method == "node_evaluation":
        return standard_evaluation(cfg, evaluation_fn=node_evaluation.main)
    elif method == "tw_evaluation":
        return standard_evaluation(cfg, evaluation_fn=tw_evaluation.main)
    elif method == "node_tw_evaluation":
        return standard_evaluation(cfg, evaluation_fn=node_tw_evaluation.main)
    elif method == "queue_evaluation":
        return queue_evaluation.main(cfg)
    elif method == "edge_evaluation":
        return standard_evaluation(cfg, evaluation_fn=edge_evaluation.main)
    else:
        raise ValueError(f"Invalid evaluation method {cfg.evaluation.used_method}")
