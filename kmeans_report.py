import argparse
import csv
import os
from contextlib import redirect_stdout
from io import StringIO

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np

from src.methods.kmeans import KMeans
from src.utils import accuracy_fn, macrof1_fn, normalize_fn


def load_classification_data(data_path, use_test=False, distance_metric="euclidean"):
    feature_data = np.load(data_path, allow_pickle=True)
    train_features = feature_data["xtrain"]
    test_features = feature_data["xtest"]
    train_labels = feature_data["ytrainclassif"].astype(int)
    test_labels = feature_data["ytestclassif"].astype(int)

    if use_test:
        xtrain, ytrain = train_features, train_labels
        xvalid, yvalid = test_features, test_labels
    else:
        num_train = int(0.8 * len(train_features))
        xtrain, ytrain = train_features[:num_train], train_labels[:num_train]
        xvalid, yvalid = train_features[num_train:], train_labels[num_train:]

    if distance_metric == "chi_square": 
        mins = np.min(xtrain, axis = 0)
        maxs = np.max(xtrain, axis = 0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1
        
        xtrain = (xtrain - mins) / ranges
        xvalid = (xvalid - mins) / ranges

    else: 
        means = np.mean(xtrain, axis=0)
        stds = np.std(xtrain, axis=0)
        stds[stds == 0] = 1

        xtrain = normalize_fn(xtrain, means, stds)
        xvalid = normalize_fn(xvalid, means, stds)

    return xtrain, ytrain, xvalid, yvalid


def run_one_experiment(xtrain, ytrain, xvalid, yvalid, k, max_iters, run, seed, distance_metric):
    np.random.seed(seed)
    model = KMeans(K=k, max_iters=max_iters, distance_metric=distance_metric)

    # KMeans prints iteration logs; keep experiment output readable.
    with redirect_stdout(StringIO()):
        train_preds = model.fit(xtrain, ytrain)
        valid_preds = model.predict(xvalid)

    return {
        "K": k,
        "Run": run,
        "Seed": seed,
        "Max Iters": max_iters,
        "Distance": distance_metric,
        "Train Acc": accuracy_fn(train_preds, ytrain),
        "Val Acc": accuracy_fn(valid_preds, yvalid),
        "Train F1": macrof1_fn(train_preds, ytrain),
        "Val F1": macrof1_fn(valid_preds, yvalid),
    }


def save_results_csv(results, output_path):
    fieldnames = ["K", "Run", "Seed", "Max Iters", "Distance", "Train Acc", "Val Acc", "Train F1", "Val F1"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def summarize_by_k(results):
    summaries = []
    for k in sorted({row["K"] for row in results}):
        rows = [row for row in results if row["K"] == k]
        summary = {"K": k}
        for metric in ["Train Acc", "Val Acc", "Train F1", "Val F1"]:
            values = np.array([row[metric] for row in rows])
            summary[f"{metric} Mean"] = np.mean(values)
            summary[f"{metric} Std"] = np.std(values)
        summaries.append(summary)
    return summaries


def plot_metrics_vs_k(summaries, output_path):
    ks = np.array([row["K"] for row in summaries])
    plt.figure(figsize=(9, 5))

    for metric, label in [
        ("Train Acc", "Train Accuracy"),
        ("Val Acc", "Validation Accuracy"),
        ("Train F1", "Train F1"),
        ("Val F1", "Validation F1"),
    ]:
        means = np.array([row[f"{metric} Mean"] for row in summaries])
        stds = np.array([row[f"{metric} Std"] for row in summaries])

        if "Acc" in metric:
            means = means / 100
            stds = stds / 100

        plt.plot(ks, means, marker="o", linewidth=1.8, label=label)
        plt.fill_between(ks, means - stds, means + stds, alpha=0.12)

    plt.xlabel("Number of clusters K")
    plt.ylabel("Score between 0 and 1")
    plt.title("K-Means performance as number of clusters changes")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_initialization_stability(results, output_path):
    plt.figure(figsize=(9, 5))
    runs = sorted({row["Run"] for row in results})

    for run in runs:
        rows = sorted([row for row in results if row["Run"] == run], key=lambda row: row["K"])
        plt.plot(
            [row["K"] for row in rows],
            [row["Val F1"] for row in rows],
            marker="o",
            linewidth=1.4,
            alpha=0.8,
            label=f"Run {run}",
        )

    plt.xlabel("Number of clusters K")
    plt.ylabel("Validation macro F1-score")
    plt.title("Random initialization stability")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_heatmap(results, output_path):
    ks = sorted({row["K"] for row in results})
    max_iters_values = sorted({row["Max Iters"] for row in results})
    grid = np.zeros((len(max_iters_values), len(ks)))

    for i, max_iters in enumerate(max_iters_values):
        for j, k in enumerate(ks):
            values = [
                row["Val F1"]
                for row in results
                if row["K"] == k and row["Max Iters"] == max_iters
            ]
            grid[i, j] = np.mean(values)

    plt.figure(figsize=(9, 5))
    image = plt.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    plt.xticks(range(len(ks)), ks)
    plt.yticks(range(len(max_iters_values)), max_iters_values)
    plt.xlabel("Number of clusters K")
    plt.ylabel("Maximum iterations")
    plt.title("Validation F1-score heatmap")
    plt.colorbar(image, label="Validation macro F1-score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_final_block(results):
    best = max(results, key=lambda row: row["Val F1"])
    print("\nK-Means:")
    print(f"Number of Clusters: {best['K']}")
    print(f"Maximum Iterations: {best['Max Iters']}")
    print(f"Distance Metric: {best['Distance']}")
    print(f"Random Initialization Run: {best['Run']}")
    print(f"Validation Accuracy: {best['Val Acc'] / 100:.3f}")
    print(f"Validation F1-score: {best['Val F1']:.3f}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    xtrain, ytrain, xvalid, yvalid = load_classification_data(args.data_path, args.test, args.distance_metric)

    ks = list(range(args.k_min, args.k_max + 1))
    runs = list(range(1, args.runs + 1))
    max_iters_values = [int(value) for value in args.max_iters.split(",")]

    results = []
    for max_iters in max_iters_values:
        for k in ks:
            for run in runs:
                row = run_one_experiment(
                    xtrain,
                    ytrain,
                    xvalid,
                    yvalid,
                    k=k,
                    max_iters=max_iters,
                    run=run,
                    seed=args.seed + run,
                    distance_metric=args.distance_metric,
                )
                results.append(row)
                print(
                    f"K={k:2d} run={run} max_iters={max_iters}: "
                    f"train_acc={row['Train Acc']:.2f}, val_acc={row['Val Acc']:.2f}, "
                    f"train_f1={row['Train F1']:.3f}, val_f1={row['Val F1']:.3f}"
                )

    csv_path = os.path.join(args.output_dir, "kmeans_results.csv")
    save_results_csv(results, csv_path)

    main_iter = max_iters_values[0]
    main_results = [row for row in results if row["Max Iters"] == main_iter]
    summaries = summarize_by_k(main_results)
    plot_metrics_vs_k(
        summaries,
        os.path.join(args.output_dir, "kmeans_metrics_vs_k.png"),
    )
    plot_initialization_stability(
        main_results,
        os.path.join(args.output_dir, "kmeans_initialization_stability.png"),
    )

    if len(max_iters_values) > 1:
        plot_heatmap(results, os.path.join(args.output_dir, "kmeans_heatmap.png"))

    print_final_block(results)
    print(f"\nSaved results to {csv_path}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="data/features.npz", 
        type=str,
    )
    parser.add_argument(
        "--output_dir", 
        default="results/kmeans", 
        type=str,
    )
    parser.add_argument(
        "--k_min", 
        default=2, 
        type=int,
    )
    parser.add_argument(
        "--k_max", 
        default=30, 
        type=int,
    )
    parser.add_argument(
        "--runs", 
        default=5, 
        type=int,
    )
    parser.add_argument(
        "--max_iters",
        default="50",
        type=str,
        help="single value such as 200, or comma-separated values such as 50,100,200",
    )
    parser.add_argument(
        "--seed", 
        default=100, 
        type=int,
    )
    parser.add_argument(
        "--test", 
        action="store_true",
    )
    parser.add_argument(
        "--distance_metric", 
        type=str, 
        default="euclidean", 
        choices=["euclidean", "chi_square", "manhattan"],
        help="euclidean / chi_square / manhattan",
    )
    args = parser.parse_args()
    main(args)
