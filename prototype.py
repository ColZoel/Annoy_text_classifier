from modules import *

def main():
    onet_classes = pd.read_csv("occupations_workathome.csv")['title']
    onet_classes = onet_classes.to_numpy()

    true_values, noisy_data = make_random_data(onet_classes, num_obs=100000)
    yhat = pipeline(
            model = "all-MiniLM-L6-v2",
            labels = onet_classes,
            data = noisy_data,
            num_trees = 100,
            batch_size = 1024,
            save_fig=True
        )

    evaluate(true_values, yhat)

if __name__ == "__main__":
    # main()
    pass
