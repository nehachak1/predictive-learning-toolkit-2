# Predictive learning toolkit 2

A machine learning project implementing Multi-Layer Perceptrons (MLP) and K-Means Clustering from scratch using NumPy. The project explores supervised and unsupervised learning techniques without relying on external machine learning frameworks.

## Features

### Multi-Layer Perceptron (MLP)
- Classification
- Regression
- Mini-batch Gradient Descent
- Backpropagation
- Configurable Hidden Layer Size

### Activation Functions
- ReLU
- Tanh
- Sigmoid
- Linear

### K-Means Clustering
- Euclidean Distance
- Manhattan Distance
- Chi-Square Distance

### Evaluation Metrics

#### Classification
- Accuracy
- Macro F1-Score

#### Regression
- Mean Squared Error (MSE)

## Project Structure

```text
.
├── src/
│   ├── methods/
│   │   ├── mlp.py
│   │   ├── kmeans.py
│   │   └── dummy_methods.py
│   │
│   ├── activations.py
│   ├── losses.py
│   └── utils.py
│
├── data/
├── main.py
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/predictive-learning-toolkit-2.git
cd predictive-learning-toolkit-2

pip install numpy
```

## Usage

### MLP Classification

```bash
python main.py \
    --task classification \
    --method mlp \
    --activation relu \
    --hidden_dim 64 \
    --lr 0.001 \
    --max_iters 1000
```

### MLP Regression

```bash
python main.py \
    --task regression \
    --method mlp \
    --activation relu \
    --hidden_dim 64
```

### K-Means Clustering

```bash
python main.py \
    --task classification \
    --method kmeans \
    --K 3 \
    --distance_metric euclidean
```

## Learning Objectives

- Neural Network Fundamentals
- Backpropagation
- Gradient-Based Optimization
- Supervised Learning
- Unsupervised Learning
- Feature Normalization
- Model Evaluation
- Distance-Based Clustering

## Technologies

- Python
- NumPy
