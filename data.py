"""
Use 1000k samples of previously scraped data to train and evaluate the model. produce Synthetic data by adding noise to the scraped data.


REDDIT DATA
Data retrieved from: https://www.reddit.com/r/datasets/comments/w340kj/dataset_of_job_descriptions_for_your_pleasure/
Data hosted at: https://drive.google.com/drive/folders/1XxNuhiei5taFR6gziofYAx0oWfGeV7y9

KAGGLE DATA
SOURCE: https://www.kaggle.com/datasets/jatinchawda/job-titles-and-description

Run this file to download the dataset from Kaggle and the Google Drive link above.
"""
import kagglehub
import shutil
import os
import pandas as pd
from glob import glob
import os.path


def sample_for_training(data, num_obs=1000):
    """Sample data for training."""
    return data.sample(num_obs)

def create_training_data(num_obs=1000):
    """Create training data from scraped data."""

    df = pd.read_parquet('data/kaggle_clean_data.parquet')
    df = df.rename(columns={"job_title": "title"})
    df1 = pd.read_parquet('data/reddit_data.parquet')
    df = pd.concat([df, df1])
    training_data = sample_for_training(df, num_obs)
    training_data.to_csv('data/training_data.csv', index=False)


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
        create_training_data(num_obs=500)
        print("Training data created.")