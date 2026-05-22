import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np

from src.activations import ReLU, Sigmoid, Tanh, Linear
from src.losses import MSE
from src.methods.mlp import MLP
from src.utils import normalize_fn, mse_fn


def load_regression_data(data_path):
    feature_data = np.load(data_path, allow_pickle=True)

    train_features = feature_data["xtrain"]
    train_labels = feature_data["ytrainreg"]

    num_train = int(0.8 * len(train_features))

    xtrain, ytrain = train_features[:num_train], train_labels[:num_train]
    xvalid, yvalid = train_features[num_train:], train_labels[num_train:]

    means = np.mean(xtrain, axis=0)
    stds = np.std(xtrain, axis=0)
    stds[stds == 0] = 1

    xtrain = normalize_fn(xtrain, means, stds)
    xvalid = normalize_fn(xvalid, means, stds)

    return xtrain, ytrain, xvalid, yvalid


def run_one_experiment(
    xtrain,
    ytrain,
    xvalid,
    yvalid,
    lr,
    max_iters,
    hidden_dim,
    batch_size,
    activation_name,
    seed,
):
    np.random.seed(seed)

    input_dim = xtrain.shape[1]

    if activation_name == "relu":
        hidden_activation = ReLU
    elif activation_name == "tanh":
        hidden_activation = Tanh
    elif activation_name == "sigmoid":
        hidden_activation = Sigmoid
    elif activation_name == "linear":
        hidden_activation = Linear
    else:
        raise ValueError("activation must be 'relu', 'tanh', 'sigmoid' or 'linear'")

    model = MLP(
        dimensions=(input_dim, hidden_dim, 1),
        activations=(hidden_activation, Linear),
    )

    ytrain_fit = ytrain.reshape(-1, 1)

    train_preds = model.fit(
        xtrain,
        ytrain_fit,
        loss=MSE,
        epochs=max_iters,
        batch_size=batch_size,
        learning_rate=lr,
    )

    train_preds = train_preds.squeeze()
    valid_preds = model.predict(xvalid).squeeze()

    return {
        "Learning Rate": lr,
        "Hidden Dim": hidden_dim,
        "Batch Size": batch_size,
        "Activation": activation_name,
        "Max Iters": max_iters,
        "Train MSE": mse_fn(train_preds, ytrain),
        "Val MSE": mse_fn(valid_preds, yvalid),
    }


def save_results_csv(results, output_path):
    fieldnames = [
        "Learning Rate",
        "Hidden Dim",
        "Batch Size",
        "Activation",
        "Max Iters",
        "Train MSE",
        "Val MSE",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_results(results, output_path):
    """
    Plot validation MSE as a function of learning rate,
    with one curve per hidden dimension.
    """

    # Find best global config to keep other params fixed
    best = min(results, key=lambda row: row["Val MSE"])

    fixed_activation = best["Activation"]
    fixed_batch_size = best["Batch Size"]
    fixed_max_iters = best["Max Iters"]

    hidden_dims = sorted(list(set(row["Hidden Dim"] for row in results)))

    plt.figure(figsize=(9, 5))

    for hidden_dim in hidden_dims:

        rows = [
            row for row in results
            if row["Hidden Dim"] == hidden_dim
            and row["Activation"] == fixed_activation
            and row["Batch Size"] == fixed_batch_size
            and row["Max Iters"] == fixed_max_iters
        ]

        rows = sorted(rows, key=lambda row: row["Learning Rate"])

        learning_rates = np.array(
            [row["Learning Rate"] for row in rows]
        )

        val_mse = np.array(
            [row["Val MSE"] for row in rows]
        )

        plt.plot(
            learning_rates,
            val_mse,
            marker="o",
            linewidth=1.8,
            label=f"Hidden Dim = {hidden_dim}",
        )

    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation MSE")

    plt.title(
        f"Validation MSE vs Learning Rate\n"
        f"(activation={fixed_activation}, "
        f"batch_size={fixed_batch_size}, "
        f"max_iters={fixed_max_iters})"
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)
    plt.close()


def print_final_block(results):
    best = min(results, key=lambda row: row["Val MSE"])

    print("\nMLP Regression:")
    print(f"Learning rate: {best['Learning Rate']}")
    print(f"Batch size: {best['Batch Size']}")
    print(f"Hidden dimension: {best['Hidden Dim']}")
    print(f"Activation: {best['Activation']}")
    print(f"Maximum iterations: {best['Max Iters']}")
    print(f"Validation MSE: {best['Val MSE']:.6f}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    xtrain, ytrain, xvalid, yvalid = load_regression_data(args.data_path)

    learning_rates = [
        float(value.strip())
        for value in args.learning_rates.split(",")
    ]

    hidden_dims = [
        int(value.strip())
        for value in args.hidden_dims.split(",")
    ]

    batch_sizes = [
        int(value.strip())
        for value in args.batch_sizes.split(",")
    ]

    max_iters_values = [
        int(value.strip())
        for value in args.max_iters.split(",")
    ]

    activations = [
        value.strip()
        for value in args.activations.split(",")
    ]

    results = []

    for activation in activations:
        for hidden_dim in hidden_dims:
            for batch_size in batch_sizes:
                for max_iters in max_iters_values:
                    for lr in learning_rates:

                        row = run_one_experiment(
                            xtrain,
                            ytrain,
                            xvalid,
                            yvalid,
                            lr=lr,
                            max_iters=max_iters,
                            hidden_dim=hidden_dim,
                            batch_size=batch_size,
                            activation_name=activation,
                            seed=args.seed,
                        )

                        results.append(row)

                        print(
                            f"activation={activation}, hidden_dim={hidden_dim}, "
                            f"batch_size={batch_size}, max_iters={max_iters}, "
                            f"lr={lr}: "
                            f"train_mse={row['Train MSE']:.6f}, "
                            f"val_mse={row['Val MSE']:.6f}"
                        )

    csv_path = os.path.join(
        args.output_dir,
        "mlp_regression_results.csv",
    )

    plot_path = os.path.join(
        args.output_dir,
        "mlp_regression_learning_rate_plot.png",
    )

    save_results_csv(results, csv_path)
    plot_results(results, plot_path)

    print_final_block(results)

    print(f"\nSaved results to {csv_path}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="results/mlp_regression",
        type=str,
    )
    parser.add_argument(
        "--learning_rates",
        default="0.00001,0.0001,0.001,0.01,0.1",
        type=str,
    )
    parser.add_argument(
        "--max_iters",
        default="50,100,200",
        type=str,
    )
    parser.add_argument(
        "--hidden_dims",
        default="8,16,32,64,128",
        type=str,
    )
    parser.add_argument(
        "--activations",
        default="relu,tanh,sigmoid,linear",
        type=str,
    )
    parser.add_argument(
        "--seed",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--batch_sizes",
        default="16,32,64",
        type=str,
    )

    args = parser.parse_args()
    main(args)
