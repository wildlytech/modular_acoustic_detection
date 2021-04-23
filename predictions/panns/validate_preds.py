import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Displays metrics for predictions")
parser.add_argument("-p", "--path_to_dataframe", help="Path to predictions df")
args = parser.parse_args()
preds = pd.read_csv(args.path_to_dataframe)

bird_names = pd.unique(preds["target"])


def top5_acc(df):
    predictions = df.preds.values
    targets = df.target.values

    acc = 0.0
    for p, t in zip(predictions, targets):
        if t in p:
            acc += 1
    # print("Top 5 Accuracy: ",acc/len(predictions))
    return acc / len(predictions)


for bird_name in bird_names:
    req = preds[preds["target"] == bird_name]
    accuracy = top5_acc(req)
    print("BIRD NAME: ", bird_name, "TOP 3 ACCURACY: ", accuracy)
