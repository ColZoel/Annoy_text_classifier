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


def create_training_data(classes, num_obs=1000):
    """Create training data from scraped data."""
    df = pd.read_parquet('data/kaggle_clean_data.parquet')
    df = df.rename(columns={"job_title": "title"})
    df1 = pd.read_parquet('data/reddit_data.parquet')
    df = pd.concat([df, df1])


def embed_batched(model, data, dim, batch_size=1000):
    """embeddings in batches."""

    num_batches = (len(data) + batch_size - 1) // batch_size
    batch_indices = np.array_split(np.arange(len(data)), num_batches)

    embeds = np.vstack([model.encode(data[indices], convert_to_numpy=True, show_progress_bar=False) for indices in batch_indices])

    return embeds


def build_tree(embeddings, num_trees=10):
    """Build an Annoy index for the given embeddings."""
    t = AnnoyIndex(embeddings.shape[1], 'euclidean')
    for i, emb in enumerate(embeddings):
        t.add_item(i, emb)
    t.build(num_trees)
    print("Annoy index built with", num_trees, "trees.")
    return t


# def predict_labels(tree, classes, embeddings, neighbors=1):
#     """Predict the nearest neighbor labels for the given embeddings."""
#     indices = np.array([tree.get_nns_by_vector(emb, neighbors) for emb in embeddings])
#     neighbor_labels = np.array([[classes[idx] for idx in idxs] for idxs in indices])
#     most_common_labels = np.array([max(set(labels), key=list(labels).count) for labels in neighbor_labels])
#     return most_common_labels

def predict_labels(tree, classes, embeddings, neighbors=1):
    """Predict the nearest neighbor labels for the given embeddings."""
    # Retrieve neighbor indices for each embedding
    indices = [tree.get_nns_by_vector(emb.tolist(), neighbors) for emb in embeddings]

    # Map indices to their corresponding class labels
    neighbor_labels = [[classes[idx] for idx in idxs] for idxs in indices]

    # Determine the most common label among neighbors
    most_common_labels = [Counter(labels).most_common(1)[0][0] for labels in neighbor_labels]

    return np.array(most_common_labels)


def evaluate(y_true, y_pred):
    """Evaluate the classification performance."""
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


def visualize(x_embeddings, x_labels, class_embeddings, class_labels, save: bool = False):
    """Visualize the embeddings using PCA."""
    x_embeddings = np.vstack(x_embeddings)
    x_labels = np.array(x_labels)
    class_embeddings = np.vstack(class_embeddings)
    class_labels = np.array(class_labels)

    pca = PCA(n_components=2)
    pca_embed = pca.fit_transform(x_embeddings)
    pca_class_embed = pca.transform(class_embeddings)

    unique_classes, counts = np.unique(x_labels, return_counts=True)
    top_20_indices = np.argsort(-counts)[:20]
    top_5_indices = np.argsort(-counts)[:5]
    unique_classes_20 = unique_classes[top_20_indices]
    unique_classes_5 = unique_classes[top_5_indices]

    colors = plt.colormaps['tab20']
    class_to_color = {cls: colors(i) for i, cls in enumerate(unique_classes_20)}

    plt.figure(figsize=(10, 10))

    # plot input embeddings
    for cls in unique_classes_20:
        cls_idx = np.where(x_labels == cls)[0]
        print(f"Class: {cls}, Count: {len(cls_idx)}")
        plt.scatter(pca_embed[cls_idx, 0], pca_embed[cls_idx, 1],
                    color=class_to_color[cls], alpha=0.6, marker='o')

    # Label classes
    for cls in unique_classes_20:
        cls_idx = np.where(class_labels == cls)[0]
        plt.scatter(pca_class_embed[cls_idx, 0], pca_class_embed[cls_idx, 1],
                    color=class_to_color[cls], label=f"{cls}", alpha=1.0, marker='x', s=250)
        # for idx in cls_idx:
        #     plt.text(pca_class_embed[idx, 0], pca_class_embed[idx, 1], cls, fontsize=9, ha='center', va='bottom')

    plt.title("Embedding Clusters in 2D: Top 20 Occupations")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    if save:
        plt.savefig("embeddings.png")
    return


def pipeline(model: str, labels: np.array, data: np.array, num_trees: int, num_neighbors: int, batch_size: int, save_fig: bool = False):

    # 1. Load pre-trained model
    model = SentenceTransformer(model)

    # 2. Encode target classes
    print(f"Encoding {len(labels)} target label values")
    target_embeddings = model.encode(labels, convert_to_numpy=True, show_progress_bar=True)

    # 3. Build Annoy Index for target classes
    print(f"Building Annoy index with {num_trees} trees")
    tree = build_tree(target_embeddings, num_trees=num_trees)

    # 4. Encode feature space and classify
    print(f"Encoding {len(data)} feature vectors")
    feature_embeddings = embed_batched(model, data, target_embeddings.shape[1], batch_size=batch_size)

    # 5. Predict labels
    print("Predicting labels")
    yhat = predict_labels(tree, labels, feature_embeddings, neighbors=num_neighbors)

    # 6. Visualize
    visualize(feature_embeddings, yhat, target_embeddings, labels, save=save_fig)

    return yhat