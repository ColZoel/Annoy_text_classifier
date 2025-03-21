import pandas as pd

from modules import *
from data import *

def main():
    onet_classes = pd.read_csv(r"data/occupations_workathome.csv")['title']
    onet_classes = onet_classes.to_numpy()

    k = pd.read_parquet("data/kaggle_clean_data.parquet")
    r = pd.read_parquet("data/reddit_data.parquet")
    k =k.rename(columns={"job_title": "title"})
    data = pd.concat([k, r]).to_numpy()
    # true_values, noisy_data = make_random_data(onet_classes, num_obs=1000000)

    yhat = pipeline(
            model="all-MiniLM-L6-v2",
            labels=onet_classes,
            data=data,
            num_trees=5000,
            num_neighbors=10,
            batch_size=1024,
            save_fig=True
        )

    # evaluate(true_values, yhat)


if __name__ == "__main__":
    main()
