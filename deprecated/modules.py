"""
Functions and tools for the occupation classification project.
"""

import time
import pandas as pd
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"(finished in {time.time() - start:.2f} seconds, "
              f"avg: {(time.time() - start) / len(args[0]):.4f} sec/label)")
        return result
    return wrapper

# ################################################## EMBEDDINGS ##################################################


@timer
def embed(model: SentenceTransformer, data: np.array):
    """Embeddings for the given data."""

    print(f"\nEncoding {len(data)} target label values")
    embeddings = model.encode(data, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


@timer
def embed_batched(model, data: np.array, batch_size: int = 1000) -> np.array:
    """embeddings in batches."""

    num_batches = (len(data) + batch_size - 1) // batch_size

    print(f"\nEncoding {len(data)} target label values ({num_batches} batches)")
    batch_indices = np.array_split(np.arange(len(data)), num_batches)

    embeddings = np.vstack([model.encode(data[indices],
                                         convert_to_numpy=True,
                                         show_progress_bar=False) for indices in batch_indices])

    return embeddings


def save_embeddings(embeddings: np.array, path: str) -> None:
    np.savez_compressed(path, **embeddings)
    return None


def load_embeddings(path: str) -> np.array:
    return np.load(path)

# ################################################### TREE  ####################################################


@timer
def build_tree(embeddings: np.array, num_trees: int = 10) -> AnnoyIndex:
    """Build an Annoy index for the given embeddings."""
    print(f"\nBuilding Annoy index with {num_trees} trees")
    t = AnnoyIndex(embeddings.shape[1], 'euclidean')
    for i, emb in enumerate(embeddings):
        t.add_item(i, emb)
    t.build(num_trees)

    return t


# ################################################### PREDICT ##################################################

@timer
def predict_labels(tree: AnnoyIndex,
                   classes: [str, np.array],
                   embeddings: np.array,
                   neighbors: int = 1) -> np.array:
    """Predict the nearest neighbor labels for the given embeddings.
    Uses majority voting to determine the most common label among the neighbors.
    :param tree: Built Annoy index
    :param classes: Class labels
    :param embeddings: Embeddings to predict
    :param neighbors: Number of neighbors to consider
    :return: Predicted labels
    """
    # Retrieve neighbor indices for each embedding

    print(f"\nPredicting labels for {len(embeddings)} embeddings")
    indices = [tree.get_nns_by_vector(emb.tolist(), neighbors) for emb in embeddings]

    # Map indices to their corresponding class labels
    neighbor_labels = [[classes[idx] for idx in idxs] for idxs in indices]

    # Determine the most common label among neighbors
    most_common_labels = [Counter(labels).most_common(1)[0][0] for labels in neighbor_labels]

    return np.array(most_common_labels)


def save_pred(x: np.array, yhat: np.array, path: str) -> None:

    df = pd.DataFrame([x, yhat], columns=['label', 'yhat'])
    df.to_csv(path)
    return


#   ################################################### EVALUATE ##################################################
def evaluate(y_true: np.array, y_pred: np.array) -> None:
    """Evaluate the classification performance.

    :param y_true: True labels
    :param y_pred: Predicted labels
    """
    comp = pd.DataFrame({"label": y_true, "yhat": y_pred})
    comp['correct'] = comp['label'] == comp['yhat']
    accuracy = comp['correct'].mean()
    precision = comp.groupby('yhat')['correct'].mean().mean()
    recall = comp.groupby('label')['correct'].mean().mean()
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"Accuracy: {accuracy:.4f}"
          f"\nPrecision: {precision:.4f}"
          f"\nRecall: {recall:.4f}"
          f"\nF1: {f1:.4f}")

    report = classification_report(y_true, y_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv("report.csv")
    return


def visualize(x_embeddings: np.array,
              x_labels: [list, np.array],
              class_embeddings: np.array,
              class_labels: [list, np.array],
              top: int = 10,
              label_points: bool = False,
              figsize=(10, 10),
              save: str = None) -> None:

    """Visualize the embeddings using PCA.
    :param x_embeddings: Embeddings for the input data
    :param x_labels: Labels for the input data
    :param class_embeddings: Embeddings for the target classes
    :param class_labels: Labels for the target classes
    :param top: Number of top classes to display
    :param label_points: Label the target class points
    :param figsize: Figure size
    :param save: Save the plot to a file
    :return: None

    """
    x_embeddings = np.vstack(x_embeddings)
    x_labels = np.array(x_labels)
    class_embeddings = np.vstack(class_embeddings)
    class_labels = np.array(class_labels)

    pca = PCA(n_components=2)
    pca_embed = pca.fit_transform(x_embeddings)
    pca_class_embed = pca.transform(class_embeddings)

    unique_classes, counts = np.unique(x_labels, return_counts=True)
    top_n_indices = np.argsort(-counts)[:top]
    n_unique_classes = unique_classes[top_n_indices]

    colors = plt.colormaps['tab20']
    class_to_color = {cls: colors(i) for i, cls in enumerate(n_unique_classes)}

    plt.figure(figsize=figsize)

    # plot input embeddings
    for cls in n_unique_classes:
        cls_idx = np.where(x_labels == cls)[0]
        print(f"Class: {cls}, Count: {len(cls_idx)}")
        plt.scatter(pca_embed[cls_idx, 0], pca_embed[cls_idx, 1],
                    color=class_to_color[cls], alpha=0.6, marker='o')

    # Label classes
    for cls in n_unique_classes:
        cls_idx = np.where(class_labels == cls)[0]
        plt.scatter(pca_class_embed[cls_idx, 0], pca_class_embed[cls_idx, 1],
                    color=class_to_color[cls], label=f"{cls}", alpha=1.0, marker='x', s=250)
        plt.title(f"Embedding Clusters: Top {top} Occupations")

        if label_points:
            for idx in cls_idx:
                plt.text(pca_class_embed[idx, 0], pca_class_embed[idx, 1], cls, fontsize=9, ha='center', va='bottom')

    if not label_points:
        plt.legend(loc='upper right')

    plt.show()
    if save is not None:
        plt.savefig(save)
    return
