"""
Modularized form of the code
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from tools.models1 import *
from typing import Union, Self


# Performance timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"(finished in {time.time() - start:.2f} seconds, "
              f"avg: {(time.time() - start) / len(args[1]):.4f} sec/label)")
        return result

    return wrapper


# ####################################################################################################
class TransformerLoader:

    """
    Tool to assist loading models. Models can be loaded in 3 ways:
    (1) directly from Hugging Face equivalent to `SentenceTransformer(model)`
    (2) from a local directory containing the model
    (3) from a remote URL pointing to a compressed model file (tarball or zip).
    The model is first saved locally from the remote URL and then loaded.

    """

    def __init__(self, model: Union[os.PathLike, str]):

        self.model = model
        self.dst = None

    def from_remote(self, url: str,
                    dst: Union[os.PathLike, str, None] = None
                    ) -> Union[os.PathLike, str, None]:

        """
        Download a model from a remote URL and save it to a local directory.
        :param url: url of the model source
        :param dst: path to save the model
        :return:
        """

        if dst is None:
            dst = os.path.join(os.getcwd(), "models")
            os.makedirs(dst, exist_ok=True)

        # Ensure the local directory exists
        elif os.path.basename(dst) != "models":
            os.makedirs(os.path.join(dst, "models"), exist_ok=True)
            dst = os.path.join(dst, "models")

        try:
            downloader = ModelDownloader(url, dst)
            downloader.check_compression()
        except ValueError as e:
            raise ValueError(f"{e}")

        # Extract the filename from the URL
        filename = urlparse(url).path.split('/')[-1]
        dst = os.path.join(dst, filename)

        # Send a GET request to the URL
        print(f"Pulling {filename}. . .")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for request errors

        # Write the content to a local file
        with open(dst, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    file.write(chunk)

        print(f"{filename} saved to {dst}")

        self.model = filename
        self.dst = dst

        return self.dst

    def search_local(self, directory: Union[os.PathLike, str, None] = None) -> AnnoyIndex:

        """
        Search a local directory for a model of the same name as the one passed to the class.
        :param directory: directory to search
        :return: The loaded model, or a list of available models if not found
        """

        if directory is None:
            directory = os.getcwd()
        models = local_models(directory)
        if self.model not in models:
            raise ValueError(f"{self.model} not found in {os.getcwd()}/models. Available models: {models}")

        self.dst = os.path.join(directory, self.model)
        return self.dst

    def load(self):
        if self.dst is not None:
            self.model = SentenceTransformer(self.dst)

        else:
            self.model = SentenceTransformer(self.model)

        return self.model


class Encoder:
    """
    Encoder for the given model. The model can be a string or a SentenceTransformer object.
    Embeddings can be saved to or loaded from disk.
    """

    def __init__(self, model: Union[os.PathLike, str, SentenceTransformer]):

        """
        :param model: path, name, SentenceTransformer object, or TransformerLoader object
        """

        self.model = SentenceTransformer(model) if not isinstance(model, SentenceTransformer) else model
        self.embeds = None
        self.labels = None

    # noinspection PyArgumentList
    @timer
    def embed(self, data: np.array, batched: bool = False, batch_size: int = 1000) -> Self:

        """
        Embed the given data using the model.
        :param data: Array of string data to be embedded
        :param batched: Whether to use batched encoding, better for large datasets
        :param batch_size: Batch size for batched encoding (irrelevant if batched is False)
        :return: Encoder object
        """

        self.labels = data

        if not batched:
            print(f"\nEncoding {len(data)} target label values")
            self.embeds = self.model.encode(data, convert_to_numpy=True, show_progress_bar=True)

            return self

        else:
            num_batches = (len(data) + batch_size - 1) // batch_size
            print(f"\nEncoding {len(data)} target label values ({num_batches} batches)")
            batch_indices = np.array_split(np.arange(len(data)), num_batches)

            self.embeds = np.vstack([self.model.encode(data[indices.tolist()],
                                                       convert_to_numpy=True,
                                                       show_progress_bar=False) for indices in batch_indices])
            return self

    def save(self, path: Union[os.PathLike, str]) -> np.array:
        np.savez_compressed(path, labels=self.labels, embeds=self.embeds)
        return self.labels, self.embeds

    def load(self, path: str) -> np.array:
        data = np.load(path)
        self.labels = data['arr_0']
        self.embeds = data['arr_1']
        self.embeds = np.load(path)
        return self.labels, self.embeds


class Classifier:

    """
    Classifier using Annoy index for fast nearest neighbor search.
    """

    def __init__(self):
        self.tree = None
        self.eval = {}
        self.x_embeddings = None
        self.x_labels = None
        self.pred = None
        self.y_embeddings = None
        self.y_labels = None

    def build_tree(self, labels: np.array, embeddings: np.array, num_trees=10):

        """Build an Annoy index for the given embeddings.
        labels should be same size as embeddings.
        :param labels: string labels
        :param embeddings: Embeddings to build the index
        :param num_trees: Number of trees to build
        """

        print(f"\nBuilding Annoy index with {num_trees} trees")
        t = AnnoyIndex(embeddings.shape[1], 'euclidean')
        for i, emb in enumerate(embeddings):
            t.add_item(i, emb)
        t.build(num_trees)
        self.tree = t

        self.y_embeddings = embeddings
        self.y_labels = labels

        return self

    def predict(self,
                labels: np.array,
                embeddings: np.array,
                neighbors: int = 1,
                save_path: Union[os.PathLike, str, None] = None) -> Self:

        """
        Predict the labels for the given embeddings using the Annoy index.
        :param labels: labels for the embeddings
        :param embeddings: numerical representation of the labels (should be same length as labels)
        :param neighbors: number of neighbors to consider
        :param save_path:
        :return:
        """

        print(f"\nPredicting labels for {len(embeddings)} embeddings")
        indices = [self.tree.get_nns_by_vector(emb.tolist(), neighbors) for emb in embeddings]

        # Map indices to their corresponding class labels
        neighbor_labels = [[labels[idx] for idx in idxs] for idxs in indices]

        # Determine the most common label among neighbors
        y_pred = [Counter(labels).most_common(1)[0][0] for labels in neighbor_labels]

        self.x_labels = labels
        self.x_embeddings = embeddings
        self.pred = y_pred

        # Save the predictions
        if save_path is not None:
            if save_path.endswith(".csv"):
                df = pd.DataFrame([labels, y_pred],
                                  columns=["label", "yhat "])
                df.to_csv(save_path)
            elif save_path.endswith(".parquet"):
                df = pd.DataFrame([labels, y_pred],
                                  columns=["label", "yhat"])
                df.to_parquet(save_path)

            else:
                np.savez_compressed(save_path, [labels, y_pred])

        return self

    def evaluate(self,
                 y_true: np.array,
                 save_path: Union[os.PathLike, str, None] = None) -> None:

        """Evaluate the classification performance against an array of ground truth labels.
        You must predict the labels of the corresponding text before calling this function.
        :param y_true: True labels
        :param save_path: Name or path of the report csv
        :return: prints overall metrics to console and saves micro/macro metrics to a csv if path is provided
        """

        # macro (overall) metrics
        comp = pd.DataFrame({"label": y_true, "yhat": self.pred})
        comp['correct'] = comp['label'] == comp['yhat']
        accuracy = comp['correct'].mean()
        precision = comp.groupby('yhat')['correct'].mean().mean()
        recall = comp.groupby('label')['correct'].mean().mean()
        f1 = 2 * (precision * recall) / (precision + recall)

        macros = {
            "overall": {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1
                        }
        }

        print(f"Accuracy: {accuracy:.4f}"
              f"\nPrecision: {precision:.4f}"
              f"\nRecall: {recall:.4f}"
              f"\nF1: {f1:.4f}")

        # Micro (per-class) metrics
        report = classification_report(y_true, self.pred, output_dict=True)
        report = {**report, **macros}
        self.eval = report
        if save_path is not None:
            pd.DataFrame(report).T.to_csv(f"{save_path}.csv")
        return

    def visualize(self,
                  top: int = 10,
                  label_points: bool = False,
                  figsize: tuple[int, int] = (10, 10),
                  save: Union[os.PathLike, None] = None) -> None:

        """
        Visualize the embeddings and their labels.
        :param top: Number of top classes to plot (>20 classes runs out of colors)
        :param label_points: Label the target class points
        :param figsize: Figure size (larger size better for large top)
        :param save: path to save the plot
        :return:
        """
        x_embeddings = np.vstack(self.x_embeddings)
        x_labels = np.array(self.pred)
        y_embeddings = np.vstack(self.y_embeddings)
        y_labels = np.array(self.y_labels)

        # PCA
        pca = PCA(n_components=2)
        pca_embed_x = pca.fit_transform(x_embeddings)
        pca_embed_y = pca.transform(y_embeddings)

        x_unique_classes, x_counts = np.unique(x_labels, return_counts=True)
        n_unique_classes = x_unique_classes[np.argsort(-x_counts)[:top]]

        # Plot
        colors = plt.colormaps['tab20']
        class_to_color = {cls: colors(i) for i, cls in enumerate(n_unique_classes)}

        plt.figure(figsize=figsize)

        # plot input embeddings
        for cls in n_unique_classes:
            cls_idx = np.where(x_labels == cls)[0]
            print(f"Class: {cls}, Count: {len(cls_idx)}")
            plt.scatter(pca_embed_x[cls_idx, 0], pca_embed_x[cls_idx, 1],
                        color=class_to_color[cls], alpha=0.6, marker='o')

        # Label classes
        for cls in n_unique_classes:
            cls_idx = np.where(y_labels == cls)[0]
            plt.scatter(pca_embed_y[cls_idx, 0], pca_embed_y[cls_idx, 1],
                        color=class_to_color[cls], label=f"{cls}", alpha=1.0, marker='x', s=250)
            plt.title(f"Embedding Clusters: Top {top} Occupations")

            if label_points:
                for idx in cls_idx:
                    plt.text(pca_embed_y[idx, 0], pca_embed_y[idx, 1], cls, fontsize=9, ha='center',
                             va='bottom')

        if not label_points:
            plt.legend(loc='upper right')

        plt.show()
        if save is not None:
            plt.savefig(save)
        return
