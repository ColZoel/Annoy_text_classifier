import pandas as pd

from modules import *
from tools.data import *


def training():
    data = pd.read_csv('data/eval.csv')
    data['title'] = clean_data(data['title'])
    data.to_csv('data/eval.csv', index=False)
    return data


def get_embeddings(model, data):

    labels = pd.read_csv('data/occupations_workathome.csv')['title'].to_numpy()
    target_embeddings = embed(model, labels)
    feature_embeddings = embed_batched(model, data['title'].to_numpy())

    save_embeddings(feature_embeddings, 'embeds/features.npz')
    save_embeddings(target_embeddings, 'embeds/target.npz')

    return target_embeddings, feature_embeddings


def pipeline(model: str, labels: np.array, data: np.array,
             num_trees: int, num_neighbors: int,
             batch_size: int):

    # 1. Load pre-trained model
    model = SentenceTransformer(model)

    # 2. Target and feature embeddings
    target_embeddings, feature_embeddings = get_embeddings(model, data)

    # 3. Build Annoy Index for target classes
    tree = build_tree(target_embeddings, num_trees=num_trees)

    # 4. Predict labels
    yhat = predict_labels(tree, labels, feature_embeddings, neighbors=num_neighbors)

    return target_embeddings, feature_embeddings, yhat





def main():
    onet_classes = pd.read_csv(r"data/occupations_workathome.csv")['title']
    onet_classes = onet_classes.to_numpy()
    y, X = read_eval_data('data/eval.csv')

    y_embed, x_embed, yhat = pipeline(
            model="all-MiniLM-L6-v2",
            labels=onet_classes,
            data=X,
            num_trees=5000,
            num_neighbors=1,
            batch_size=1024,
        )

    visualize(x_embed, yhat, y_embed, onet_classes,
              top=5,
              label_points=True,
              figsize=(6, 6),
              save="100_small_samples.png")

    save_pred(X, yhat, "small_samples.csv")

    evaluate(y, yhat)


if __name__ == "__main__":
    main()
