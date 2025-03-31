from modules2 import *
from tools.data import make_random_data

# 0. randomly generate data
true, data = make_random_data(classes="data/occupations_workathome.csv", num_obs=10, colidx=1)

# 1. Embed the data
encoder = Encoder("all-MiniLM-L6-v2")
# y_labels, y_embeddings = encoder.lo
y_labels, y_embeddings = encoder.embed(true).save("embeds/occupations.npz")
x_labels, x_embeddings = encoder.embed(data, batched=True, batch_size=1024).save("embeds/test.npz")

# 2. Build the tree
tree = Classifier().build_tree(y_labels, y_embeddings, num_trees=5)

# 3. Predict the labels
predicted = tree.predict(x_labels, x_embeddings)

# 4. Evaluate the model
predicted.evaluate(true)

# 6. Visualize the data
predicted.visualize(top=5, label_points=True)