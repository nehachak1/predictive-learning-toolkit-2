import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE
from src.activations import Sigmoid, ReLU
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, label_to_onehot, onehot_to_label, get_n_classes
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    if args.task == "classification":
        train_labels = train_labels_classif.astype(int)
        test_labels = test_labels_classif.astype(int)
    elif args.task == "regression":
        train_labels = train_labels_reg
        test_labels = test_labels_reg
    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        num_train = int(0.8 * len(train_features))
        xtest, ytest = train_features[num_train:], train_labels[num_train:]
        xtrain, ytrain = train_features[:num_train], train_labels[:num_train]
    else:
        xtrain, ytrain = train_features, train_labels
        xtest, ytest = test_features, test_labels

    ### WRITE YOUR CODE HERE to do any other data processing
    means = np.mean(xtrain, axis=0)
    stds = np.std(xtrain, axis=0)
    stds[stds == 0] = 1

    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters, distance_metric=args.distance_metric)

    elif args.method == "mlp":
        input_dim = xtrain.shape[1]
        if args.task == "classification":
            output_dim = get_n_classes(ytrain)
            method_obj = MLP(
                dimensions=(input_dim, 32, output_dim),
                activations=(ReLU, Sigmoid),
            )
        else:
            method_obj = MLP(
                dimensions=(input_dim, 32, 1),
                activations=(ReLU, ReLU),
            )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":

        if args.method == "mlp":
            ytrain_fit = label_to_onehot(ytrain, C=get_n_classes(ytrain))
            preds_train = method_obj.fit(
                xtrain,
                ytrain_fit,
                loss=MSE,
                epochs=args.max_iters,
                batch_size=32,
                learning_rate=args.lr,
            )
            preds_train = onehot_to_label(preds_train)
            preds = onehot_to_label(method_obj.predict(xtest))
        else:
            preds_train = method_obj.fit(xtrain, ytrain)
            preds = method_obj.predict(xtest)

        acc = accuracy_fn(preds_train, ytrain)
        macro_f1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macro_f1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macro_f1 = macrof1_fn(preds, ytest)
        print(f"{'Test' if args.test else 'Validation'} set: accuracy = {acc:.3f}% - F1-score = {macro_f1:.6f}")

    elif args.task == "regression":
        assert args.method != "kmeans", f"You should use kmeans as a classification method"
        assert args.method == "mlp", f"You should use mlp as a regression method"

        ytrain_fit = ytrain.reshape(-1, 1)
        preds_train = method_obj.fit(
            xtrain,
            ytrain_fit,
            loss=MSE,
            epochs=args.max_iters,
            batch_size=32,
            learning_rate=args.lr,
        )
        preds_train = preds_train.squeeze()
        preds = method_obj.predict(xtest).squeeze()

        mse = mse_fn(preds_train, ytrain)
        print(f"\nTrain set: MSE = {mse:.6f}")

        mse = mse_fn(preds, ytest)
        print(f"{'Test' if args.test else 'Validation'} set: MSE = {mse:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    parser.add_argument(
        "--distance_metric", 
        type=str, 
        default="euclidean", 
        help="euclidean / manhattan",
    )
    # Feel free to add more arguments here if you need!

    args = parser.parse_args()
    main(args)
