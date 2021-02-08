import argparse
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from preprocess_utils import read_file, get_select_vector, import_dataframes
from sklearn.metrics import classification_report
from tensorflow.compat.v1.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.compat.v1.keras.models import model_from_json, Sequential
from tensorflow.compat.v1.keras.optimizers import Adam
from youtube_audioset import get_recursive_sound_names

params_path = "param_file.json"
config_path = "model_configs/binary_relevance_model/binary_relevance_model_Wild_cfg.json"

np.random.seed(10)
random.seed(10)
tf.compat.v1.set_random_seed(10)


def create_keras_model():
    """
    Creating a Model
    """
    model = Sequential()
    model.add(Conv1D(500, input_shape=(1280, 1), kernel_size=128, strides=128, activation='relu', padding='same'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    return model


def load_model(CONFIG_DATA):
    if CONFIG_DATA["networkCfgJson"] is None:
        MODEL = create_keras_model()
    else:
        json_file = open(CONFIG_DATA["networkCfgJson"], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        MODEL = model_from_json(loaded_model_json)

        MODEL.load_weights(CONFIG_DATA["train"]["outputWeightFile"])

    return MODEL


if __name__ == "__main__":
    DESCRIPTION = "THIS SCRIPT TAKES A CONFIG AND PARAM FILE AS INPUT AND GENERATES RESULTS FOR ALL PARAMS"
    ARGUMENT_PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    ARGUMENT_PARSER.add_argument("-params_path", "--params_path", help="Path to params file", required=True)
    ARGUMENT_PARSER.add_argument("-config_path", "--config_path", help="Path to config file", required=True)

    PARSED_ARGS = ARGUMENT_PARSER.parse_args()
    CONFIG_DATA = read_file(PARSED_ARGS.config_path)
    params_file = read_file(PARSED_ARGS.params_path)

    FULL_NAME = CONFIG_DATA["aggregatePositiveLabelName"] + 'vs' + CONFIG_DATA["aggregateNegativeLabelName"]
    assert (CONFIG_DATA["ontology"]["useYoutubeAudioSet"])

    # List of paths to json files that will be used to extend
    # existing youtube ontology
    ontologyExtFiles = CONFIG_DATA["ontology"]["extension"]

    # If single file or null, then convert to list
    if ontologyExtFiles is None:
        ontologyExtFiles = []
    elif type(ontologyExtFiles) != list:
        ontologyExtFiles = [ontologyExtFiles]

    # Grab all the positive labels
    POSITIVE_LABELS = get_recursive_sound_names(CONFIG_DATA["positiveLabels"], "./", ontologyExtFiles)

    # If negative labels were provided, then collect them
    # Otherwise, assume all examples that are not positive are negative
    if CONFIG_DATA["negativeLabels"] is None:
        NEGATIVE_LABELS = None
    else:
        # Grab all the negative labels
        NEGATIVE_LABELS = get_recursive_sound_names(CONFIG_DATA["negativeLabels"], "./", ontologyExtFiles)
        # Make sure there is no overlap between negative and positive labels
        NEGATIVE_LABELS = NEGATIVE_LABELS.difference(POSITIVE_LABELS)

    DF_TRAIN, DF_TEST = \
        import_dataframes(dataframe_file_list=CONFIG_DATA["train"]["inputDataFrames"],
                          positive_label_filter_arr=POSITIVE_LABELS,
                          negative_label_filter_arr=NEGATIVE_LABELS,
                          validation_split=CONFIG_DATA["train"]["validationSplit"])

    LABELS_BINARIZED_TRAIN = pd.DataFrame()
    LABELS_BINARIZED_TRAIN[FULL_NAME] = 1.0 * get_select_vector(DF_TRAIN, POSITIVE_LABELS)

    LABELS_BINARIZED_TEST = pd.DataFrame()
    LABELS_BINARIZED_TEST[FULL_NAME] = 1.0 * get_select_vector(DF_TEST, POSITIVE_LABELS)

    TOTAL_TRAIN_TEST_EXAMPLES = LABELS_BINARIZED_TRAIN.shape[0] + LABELS_BINARIZED_TEST.shape[0]
    TOTAL_TRAIN_TEST_POSITIVE_EXAMPLES = (LABELS_BINARIZED_TRAIN[FULL_NAME] == 1).sum() + \
                                         (LABELS_BINARIZED_TEST[FULL_NAME] == 1).sum()
    TOTAL_TRAIN_TEST_NEGATIVE_EXAMPLES = (LABELS_BINARIZED_TRAIN[FULL_NAME] == 0).sum() + \
                                         (LABELS_BINARIZED_TEST[FULL_NAME] == 0).sum()
    print("NUMBER EXAMPLES (TOTAL/POSITIVE/NEGATIVE):", \
          TOTAL_TRAIN_TEST_EXAMPLES, "/", \
          TOTAL_TRAIN_TEST_POSITIVE_EXAMPLES, "/", \
          TOTAL_TRAIN_TEST_NEGATIVE_EXAMPLES)

    print("PERCENT POSITIVE EXAMPLES:",
          "{0:.2f}%".format(100.0 * TOTAL_TRAIN_TEST_POSITIVE_EXAMPLES / TOTAL_TRAIN_TEST_EXAMPLES))

    X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
    X_TRAIN_STANDARDIZED = X_TRAIN / 255
    X_TEST = np.array(DF_TEST.features.apply(lambda x: x.flatten()).tolist())

    CLF2_TRAIN = X_TRAIN.reshape((-1, 1280, 1))
    CLF2_TEST = X_TEST.reshape((-1, 1280, 1))
    CLF2_TRAIN_TARGET = LABELS_BINARIZED_TRAIN.values
    CLF2_TEST_TARGET = LABELS_BINARIZED_TEST.values

    TRAIN_TARGET_POSITIVE_PERCENTAGE = CLF2_TRAIN_TARGET.mean()

    print("TRAIN NUMBER EXAMPLES (TOTAL/POSITIVE/NEGATIVE):", \
          CLF2_TRAIN_TARGET.shape[0], "/", \
          (CLF2_TRAIN_TARGET == 1).sum(), "/", \
          (CLF2_TRAIN_TARGET == 0).sum())
    print("TRAIN PERCENT POSITIVE EXAMPLES:", "{0:.2f}%".format(100 * TRAIN_TARGET_POSITIVE_PERCENTAGE))

    if TRAIN_TARGET_POSITIVE_PERCENTAGE > 0.5:
        CLASS_WEIGHT_0 = TRAIN_TARGET_POSITIVE_PERCENTAGE / (1 - TRAIN_TARGET_POSITIVE_PERCENTAGE)
        CLASS_WEIGHT_1 = 1
    else:
        CLASS_WEIGHT_0 = 1
        CLASS_WEIGHT_1 = (1 - TRAIN_TARGET_POSITIVE_PERCENTAGE) / TRAIN_TARGET_POSITIVE_PERCENTAGE

    MODEL = load_model(CONFIG_DATA)

    for i in range(params_file["nParams"]):
        MODEL.compile(loss=params_file["loss"][i],
                      optimizer=Adam(lr=params_file["lr"][i], epsilon=params_file["epsilon"][i]),
                      metrics=['accuracy'])
        MODEL.fit(CLF2_TRAIN, CLF2_TRAIN_TARGET,
                  epochs=params_file["epochs"][i],
                  class_weight={0: CLASS_WEIGHT_0, 1: CLASS_WEIGHT_1},
                  verbose=1)

        preds = MODEL.predict(CLF2_TEST)
        preds = np.round(preds)
        print("Parameters: lr=", params_file["lr"][i], "\t",
              "loss=", params_file["loss"][i], "\t",
              "epsilon=", params_file["epsilon"][i], "\t",
              "epochs: ", params_file["epochs"][i])
        print(classification_report(CLF2_TEST_TARGET, preds))
