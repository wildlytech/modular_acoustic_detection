import argparse
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

#############################################################################
# Description and help
#############################################################################

DESCRIPTION = "Analyzes DataFrame pickle file and outputs relevant information"

#############################################################################
# Parse the input arguments given from command line
#############################################################################

ARGUMENT_PARSER = argparse.ArgumentParser(description=DESCRIPTION)
OPTIONAL_NAMED = ARGUMENT_PARSER._action_groups.pop()
REQUIRED_NAMED = ARGUMENT_PARSER.add_argument_group('required arguments')
REQUIRED_NAMED.add_argument('-f', '--filepath',
                            help='Input dataframe .pkl file',
                            required=True)
ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
PARSED_ARGS = ARGUMENT_PARSER.parse_args()

filepath = PARSED_ARGS.filepath

#############################################################################

with open(filepath, 'rb') as file_obj:
    df = pickle.load(file_obj)

if "dBFS" not in df.columns:
    print("DataFrame does not contain dBFS column")
else:
    fig = plt.figure()
    (-df.dBFS.replace(-np.inf,1)).hist()
    plt.xlabel("-dBFS")
    plt.ylabel("Number of clips/examples")
    plt.title("Distribution of decibel level")
    outputFile = filepath+"_dBFS_Dist.png"
    plt.savefig(outputFile, dpi=fig.dpi)
    plt.close(fig)
    plt.clf()
    print("Created", outputFile)

if ("start_seconds" in df.columns) and ("end_seconds" in df.columns):
    duration = df.end_seconds - df.start_seconds
elif "features" in df.columns:
    duration = df.features.apply(lambda x: len(x))
else:
    duration = None
    print("DataFrame does not contain time counts")

if duration is not None:
    fig = plt.figure()
    duration.hist()
    plt.xlabel("Clip duration")
    plt.ylabel("Number of clips/examples")
    plt.title("Distribution of audio length")
    outputFile = filepath+"_AudioLength_Dist.png"
    plt.savefig(outputFile, dpi=fig.dpi)
    plt.close(fig)
    plt.clf()
    print("Created", outputFile)

if "labels_name" not in df.columns:
    print("DataFrame does not contain labels")
else:
    fig = plt.figure()
    df.labels_name.apply(lambda x: len(x)).hist()
    plt.xlabel("Number of labels in clip")
    plt.ylabel("Number of clips/examples")
    plt.title("Distribution of label count")
    outputFile = filepath+"_NumLabels_Dist.png"
    plt.savefig(outputFile, dpi=fig.dpi)
    plt.close(fig)
    plt.clf()
    print("Created", outputFile)

    binarized_labels = pd.DataFrame()
    setOfLabels = list(set(chain.from_iterable(df.labels_name)))
    for label_name in setOfLabels:
        binarized_labels[label_name] = df.labels_name.apply(lambda x: label_name in x)
    label_percentages = binarized_labels.mean().sort_values(ascending=False)
    outputFile = filepath+"_label_percent.csv"
    label_percentages.to_csv(outputFile)
    print("Created", outputFile)

    label_combo_percentages = {}
    for index, label_name1 in enumerate(setOfLabels):
        for label_name2 in setOfLabels[(index+1):]:
            combo_percentage = (binarized_labels[label_name1]&binarized_labels[label_name2]).mean()

            if combo_percentage > 0:
                label_combo_percentages['['+label_name1 + ']+[' + label_name2 + ']'] = combo_percentage

    if len(label_combo_percentages) == 0:
        print("All examples have only one label, so there are no label combos to report")
    else:
        label_combo_percentages = pd.Series(label_combo_percentages).sort_values(ascending=False)
        outputFile = filepath+"_label_combo_percent.csv"
        label_combo_percentages.to_csv(outputFile)
        print("Created", outputFile)
