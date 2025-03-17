import random
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# -------------------------------
# Step 1: Define target ONET occupation classes
# -------------------------------
onet_classes = pd.read_csv("occupations_workathome.csv")['title']


# -------------------------------
# Step 2: Generate synthetic data (100,000 observations)
# -------------------------------
def noisify(occupation):
    """Introduce simple noise to a base occupation title."""

    # Occasionally append a random seniority level
    if random.random() < 0.3:
        occupation = occupation + " " + random.choice(["Senior", "Junior", "Lead"])

    # missing letters
    if random.random() < 0.3:
        locc = list(occupation)
        locc.pop(random.randint(0, len(occupation)-1))
        occupation = "".join(locc)

    # lowercase
    if random.random() < 0.5:
        occupation = occupation.lower()
    # all caps
    if random.random() < 0.1:
        occupation = occupation.upper()

    return occupation


# Make ground truth for evaluation
num_obs = 10000
obs = pd.DataFrame([random.choice(onet_classes) for _ in range(num_obs)], columns=["occupation"])
obs['x'] = obs['occupation'].apply(noisify)


# -------------------------------
# Step 3: Encode using All-MiniLM
# -------------------------------
# Load the pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode ONET classes
print("Encoding ONET classes...")
onet_embeddings = model.encode(onet_classes, convert_to_numpy=True, show_progress_bar=True)

# -------------------------------
# Step 4: Build the Annoy index for ONET classes
# -------------------------------
embedding_dim = onet_embeddings.shape[1]
annoy_index = AnnoyIndex(embedding_dim, 'angular')
for i, emb in enumerate(onet_embeddings):
    annoy_index.add_item(i, emb)
num_trees = 10
annoy_index.build(num_trees)
print("Annoy index built with", num_trees, "trees.")

# -------------------------------
# Step 5: Encode observations and classify via Annoy
# -------------------------------
batch_size = 1024
predicted_labels = []
start_time = time.time()


# embeddings for visualization
embeds = np.empty((0, embedding_dim))

print("Processing observations in batches...")
for i in range(0, num_obs, batch_size):
    batch_texts = obs['x'][i:i+batch_size]
    batch_embeddings = model.encode(batch_texts.to_list(), convert_to_numpy=True, show_progress_bar=False)
    embeds = np.vstack([embeds, batch_embeddings])
    # For each embedding, query the Annoy index to find the nearest ONET occupation
    for emb in batch_embeddings:
        idx = annoy_index.get_nns_by_vector(emb, 1)[0]  # Get the nearest neighbor index
        predicted_labels.append(onet_classes[idx])

end_time = time.time()


obs['yhat'] = predicted_labels
obs['correct'] = obs['occupation'] == obs['yhat']

# -------------------------------
# Results
# -------------------------------
print(f"\nProcessed {num_obs} observations in {end_time - start_time:.2f} seconds.")


# -------------------------------
# Step 6: Evaluate
# -------------------------------

# Evaluate the accuracy
accuracy = obs['correct'].mean()
precision = obs.groupby('yhat')['correct'].mean().mean()
recall = obs.groupby('occupation')['correct'].mean().mean()
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy:.4f}"
      f"\nPrecision: {precision:.4f}"
      f"\nRecall: {recall:.4f}"
      f"\nF1: {f1:.4f}")


report = classification_report(obs['occupation'], obs['yhat'], output_dict=True)
report = pd.DataFrame(report).to_csv("report.csv")

# print("\nClassification Report:", classification_report(obs['occupation'], obs['yhat']))

# -------------------------------
# Step 7: Visualize embeddings using PCA


# Convert embeddings to a NumPy array
embeddings = np.vstack(embeds)

# Reduce dimensions to 2D
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
reduced_embeddings =np.column_stack((obs['occupation'], reduced_embeddings))
# Plot clusters
plt.figure(figsize=(10, 6))

# Get unique classes and assign a color to each
# unique_classes = obs['occupation'].unique()
unique_classes = obs['yhat'].value_counts().nlargest(20).index
colors = plt.colormaps['tab20']


# Create a dictionary to map each class to a color
class_to_color = {cls: colors(i) for i, cls in enumerate(unique_classes)}

# Plot each class with its corresponding color
for cls in unique_classes:
    cls_mask = reduced_embeddings[:, 0] == cls
    plt.scatter(reduced_embeddings[cls_mask, 1], reduced_embeddings[cls_mask, 2],
                color=class_to_color[cls], label=cls, alpha=0.5)


# plt.scatter(reduced_embeddings[:, 1], reduced_embeddings[:, 2], alpha=0.5)

plt.title("2D Projection of Embeddings using PCA")
plt.show()



print("done.")
