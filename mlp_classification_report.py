import argparse
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np

from src.activations import ReLU, Sigmoid, Tanh
from src.losses import MSE
from src.methods.mlp import MLP
from src.utils import(
    accuracy_fn,
    macrof1_fn,
    normalize_fn,
    label_to_onehot, 
    onehot_to_label, 
    get_n_classes,
)

def load_classification_data(data_path):
    feature_data = np.load(data_path, allow_pickle=True)

    train_features = feature_data["xtrain"]
    train_labels = feature_data["ytrainclassif"].astype(int)

    num_train = int(0.8 * len(train_features))
    xtrain, ytrain = train_features[:num_train], train_labels[:num_train]
    xvalid, yvalid = train_features[num_train:], train_labels[num_train:]

    means = np.mean(xtrain, axis=0)
    stds = np.std(xtrain, axis=0)
    stds[stds == 0] = 1

    xtrain = normalize_fn(xtrain, means, stds)
    xvalid = normalize_fn(xvalid, means, stds)

    return xtrain, ytrain, xvalid, yvalid


def run_one_experiment(xtrain, ytrain, xvalid, yvalid, lr, max_iters, hidden_dim, batch_size, activation_name, seed):
    np.random.seed(seed)

    input_dim = xtrain.shape[1]
    output_dim = get_n_classes(ytrain)

    if activation_name == "relu":
        hidden_activation = ReLU
    elif activation_name == "tanh":
        hidden_activation = Tanh
    elif activation_name == "sigmoid": 
        hidden_activation = Sigmoid
    else:
        raise ValueError("activation must be 'relu', 'tanh' or 'sigmoid'")

    model = MLP(
        dimensions=(input_dim, hidden_dim, output_dim),
        activations=(hidden_activation, Sigmoid),
    )

    ytrain_onehot = label_to_onehot(ytrain, C=output_dim)

    train_preds = model.fit(
        xtrain,
        ytrain_onehot,
        loss = MSE,
        epochs = max_iters,
        batch_size = batch_size,
        learning_rate = lr,
    )

    train_preds = onehot_to_label(train_preds)
    valid_preds = onehot_to_label(model.predict(xvalid))

    return {
        "Learning Rate": lr,
        "Hidden Dim": hidden_dim,
        "Batch Size": batch_size,
        "Activation": activation_name,
        "Max Iters": max_iters,
        "Train Acc": accuracy_fn(train_preds, ytrain),
        "Val Acc": accuracy_fn(valid_preds, yvalid),
        "Train F1": macrof1_fn(train_preds, ytrain),
        "Val F1": macrof1_fn(valid_preds, yvalid),
    }


def save_results_csv(results, output_path):
    fieldnames = [
        "Learning Rate",
        "Hidden Dim",
        "Batch Size", 
        "Activation",
        "Max Iters",
        "Train Acc",
        "Val Acc",
        "Train F1",
        "Val F1",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(results)

def plot_results(results, output_path):
    best = max(results, key=lambda row: row["Val F1"])

    rows = [
        row for row in results
        if row["Hidden Dim"] == best["Hidden Dim"]
        and row["Batch Size"] == best["Batch Size"]
        and row["Activation"] == best["Activation"]
        and row["Max Iters"] == best["Max Iters"]
    ]

    rows = sorted(rows, key=lambda row: row["Learning Rate"])

    learning_rates = np.array([row["Learning Rate"] for row in rows])
    train_acc = np.array([row["Train Acc"] / 100 for row in rows])
    val_acc = np.array([row["Val Acc"] / 100 for row in rows])
    train_f1 = np.array([row["Train F1"] for row in rows])
    val_f1 = np.array([row["Val F1"] for row in rows])

    plt.figure(figsize=(9, 5))

    plt.plot(learning_rates, train_acc, marker="o", linewidth=1.8, label="Train Accuracy")
    plt.plot(learning_rates, val_acc, marker="o", linewidth=1.8, label="Validation Accuracy")
    plt.plot(learning_rates, train_f1, marker="o", linewidth=1.8, label="Train F1")
    plt.plot(learning_rates, val_f1, marker="o", linewidth=1.8, label="Validation F1")

    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Score between 0 and 1")
    plt.title("MLP Classification performance as learning rate changes")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def print_final_block(results):
    best = max(results, key=lambda row: row["Val F1"])

    print("\nMLP Classification:")
    print(f"Learning rate: {best['Learning Rate']}")
    print(f"Batch size: {best['Batch Size']}")
    print(f"Hidden dimension: {best['Hidden Dim']}")
    print(f"Activation: {best['Activation']}")
    print(f"Maximum iterations: {best['Max Iters']}")
    print(f"Validation accuracy: {best['Val Acc']:.3f}%")
    print(f"Validation F1-score: {best['Val F1']:.3f}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    xtrain, ytrain, xvalid, yvalid = load_classification_data(args.data_path)

    learning_rates = [float(value.strip()) for value in args.learning_rates.split(",")]
    hidden_dims = [int(value.strip()) for value in args.hidden_dims.split(",")]
    batch_sizes = [int(value.strip()) for value in args.batch_sizes.split(",")]
    max_iters_values = [int(value.strip()) for value in args.max_iters.split(",")]
    activations = [value.strip() for value in args.activations.split(",")]

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
                            lr = lr,
                            max_iters = max_iters,
                            hidden_dim = hidden_dim,
                            batch_size = batch_size,
                            activation_name = activation,
                            seed = args.seed,
                        )

                        results.append(row)

                        print(
                            f"activation={activation}, hidden_dim={hidden_dim}, "
                            f"batch_size={batch_size}, max_iters={max_iters}, lr={lr}: "
                            f"train_acc={row['Train Acc']:.2f}, val_acc={row['Val Acc']:.2f}, "
                            f"train_f1={row['Train F1']:.3f}, val_f1={row['Val F1']:.3f}"
                        )

    csv_path = os.path.join(args.output_dir, "mlp_classification_results.csv")
    plot_path = os.path.join(args.output_dir, "mlp_learning_rate_plot.png")

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
        default="results/mlp_classification",
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
        default="relu,tanh,sigmoid",
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