"""
Functions and tools for the occupation classification project.
"""

import random
import time
import pandas as pd
import numpy as np
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

    embeds = np.empty((0, dim))

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeds = np.vstack([embeds, batch_embeddings])

    return embeds


def predict_labels(tree, classes, embeddings):
    """Predict the nearest neighbor labels for the given embeddings."""
    labels = np.array([])
    for emb in embeddings:
        idx = tree.get_nns_by_vector(emb, 1)[0]
        labels = np.append(labels, classes[idx])

    return labels


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


def visualize(embeddings, labels, save: bool = False):
    """Visualize the embeddings using PCA."""
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    pca = PCA(n_components=2)
    pca_embed = pca.fit_transform(embeddings)
    # pca_embed = np.column_stack((labels, pca_embed))
    pca_embed[:, 1:] = pca_embed[:, 1:].astype(float)

    unique_classes, counts = np.unique(labels, return_counts=True)
    top_20_indices = np.argsort(-counts)[:20]
    unique_classes = unique_classes[top_20_indices]

    colors = plt.colormaps['tab20']
    class_to_color = {cls: colors(i) for i, cls in enumerate(unique_classes)}

    plt.figure(figsize=(10, 6))

    for cls in unique_classes:
        cls_idx = np.where(labels == cls)[0]

        plt.scatter(pca_embed[cls_idx, 0], pca_embed[cls_idx, 1],
                    color=class_to_color[cls], label=cls, alpha=0.5)

    plt.title("Embedding Clusters in 2D: Top 20 Occupations")
    plt.show()
    if save:
        plt.savefig("embeddings.png")
    return


def pipeline(model: str, labels: np.array, data: np.array, num_trees: int, batch_size: int, save_fig: bool = False):

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
    yhat = predict_labels(tree, labels, feature_embeddings)

    # 6. Visualize
    visualize(feature_embeddings, yhat, save=save_fig)

    return yhat