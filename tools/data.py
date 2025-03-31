"""
Use 100k samples of previously scraped data to train and evaluate the model. produce Synthetic data by adding noise to the scraped data.


REDDIT DATA
Data retrieved from: https://www.reddit.com/r/datasets/comments/w340kj/dataset_of_job_descriptions_for_your_pleasure/
Data hosted at: https://drive.google.com/drive/folders/1XxNuhiei5taFR6gziofYAx0oWfGeV7y9

KAGGLE DATA
SOURCE: https://www.kaggle.com/datasets/jatinchawda/job-titles-and-description

Run this file to download the dataset from Kaggle.
"""
import kagglehub
import shutil
import os
import random
import pandas as pd
from glob import glob
import numpy as np


def clean_data(data):
    """
    Standardize data for english ASCII-only characters.
    :param data:
    :return:
    """

    if isinstance(data, pd.DataFrame):
        df = data
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=['title'])
    if isinstance(data, pd.Series):
        df = data.to_frame()

   # non-ascii
    df['title'] = df['title'].str.encode('ascii', 'ignore').str.decode('ascii')
    df['title'] = df['title'].str.replace(r'[^\w\s]', '', regex=True)  # remove punctuation
    df['title'] = df['title'].str.replace(r'\d+', '', regex=True)  # remove digits
    df['title'] = title_casing(df['title'])  # title casing
    df['title'] = parentheses(df['title'])  # remove parentheses

    # remove empty strings
    df = df[df['title'] != ""]

    return df.title.to_numpy()


def title_casing(data):
    """
    Standardize casing to each word in the title.
    """
    return data.str.title()


def parentheses(data):
    """
    Remove parentheses from the data.
    """
    return data.str.replace(r'\(.*\)', '', regex=True).str.strip()


def create_eval_data(num_obs=None):
    """Create training data from scraped data."""

    df = pd.read_parquet('data/kaggle_clean_data.parquet')
    df = df.rename(columns={"job_title": "title"})
    df1 = pd.read_parquet('data/reddit_data.parquet')
    df = pd.concat([df, df1])
    df['label'] = ""

    if num_obs:
        df = df.sample(num_obs)

    df['title'] = clean_data(df['title'])

    df.to_csv('data/eval.csv', index=False)


def read_eval_data(path):
    """Read the evaluation data."""
    df = pd.read_csv(path)
    y = df['label'].to_numpy()
    X = df['title'].to_numpy()
    return y, X


def noisify(truevals: np.array):
    """Add noise to the data."""

    df = pd.DataFrame(truevals, columns=['X'])
    if random.random() < 0.3:
        df['X'] = df['X'].str.upper()
    if random.random() > 0.5:
        df['X'] = df['X'].str.lower()
    if random.random() < 0.5:
        df['X'] = df['X'].apply(lambda x: x + " " + random.choice(["Senior", "Junior", "Lead"]))
    if random.random() < 0.3:
        df['X'] = df['X'].apply(lambda x: "".join(list(x).pop(random.randint(0, len(x) - 1))))
    if random.random() < 0.1:
        df['X'] = df['X'].str[::-1]
    if random.random() < 0.2:
        df['X'] = df['X'].apply(lambda x: x.replace(" ", random.choice(["_", "-", ""])))
    if random.random() < 0.2:
        df['X'] = df['X'].apply(lambda x: x + str(random.randint(0, 99)))
    if random.random() < 0.2:
        df['X'] = df['X'].apply(lambda x: x[:random.randint(1, len(x))])
    return df.X.to_numpy()


def make_random_data(classes, num_obs=1000, colidx=0):
    """Create random data with noise."""
    if isinstance(classes, str) and classes.endswith('.csv'):
        df = pd.read_csv(classes)
        classes = df.iloc[:, colidx].unique()

    true_values = np.random.choice(classes, num_obs)
    noisy_data = noisify(true_values)
    return true_values, noisy_data


# Main function
if __name__ == "__main__":

    # Download the Kaggle dataset
    if not os.path.exists("data/kaggle_clean_data.parquet"):
        os.makedirs("data", exist_ok=True)
        path = kagglehub.dataset_download("jatinchawda/job-titles-and-description")
        print("Path to dataset files:", path)
        shutil.move(f"{path}/clean_data.parquet", "data/kaggle_clean_data.parquet")

        # Save only the title column
        df = pd.read_parquet("data/kaggle_clean_data.parquet", columns=["job_title"])
        df.to_parquet("data/kaggle_clean_data.parquet")

    else:
        print("Kaggle data already downloaded.")

    # Download the Reddit dataset
    if not os.path.exists("data/reddit_data.parquet"):

        if not os.path.exists("data/reddit_jobs"):
            raise FileNotFoundError("Download the Reddit dataset from the drive at"
                                    " https://drive.google.com/drive/folders/1XxNuhiei5taFR6gziofYAx0oWfGeV7y9 ."
                                    "\nSave as data/reddit_jobs.")

        print("Formatting Reddit data...")
        df = pd.concat([pd.read_csv(file) for file in glob(f"data/reddit_jobs/*.csv")])
        df=df['title'].to_frame()
        df.to_parquet("data/reddit_data.parquet")
        print("Reddit data saved to data/reddit_data.parquet")
    else:
        print("Reddit data already formatted correctly.")

    # Create training data
    if not os.path.exists("data/training_data.csv"):
        create_eval_data(num_obs=500)
        print("Training data created.")