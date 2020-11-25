"""
Traning a Mulit-label Model
"""
import sys
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, hamming_loss
from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.compat.v1.keras.optimizers import Adam
from youtube_audioset import get_recursive_sound_names, get_all_sound_names
from youtube_audioset import EXPLOSION_SOUNDS, MOTOR_SOUNDS, WOOD_SOUNDS, \
                             HUMAN_SOUNDS, NATURE_SOUNDS, DOMESTIC_SOUNDS, TOOLS_SOUNDS
import balancing_dataset
from pprint import pprint
import os
import json
from colorama import Fore,Style

cfg_path = "/home/madlad/Code_Projects/modular_acoustic_detection/model_configs/multilabel_model/multilabel_maxpool.json"
def read_config(filepath):
    with open(filepath) as f:
        config = json.load(f)
    return config

config = read_config(cfg_path)


output_wt_file = config["train"]["outputWeightFile"]
pathToFileDirectory = "/".join(output_wt_file.split('/')[:-1]) + '/'
if not os.path.exists(pathToFileDirectory):
  os.makedirs(pathToFileDirectory)

assert(config["ontology"]["useYoutubeAudioSet"])




'''
########################################################################
          # Get all sound names
########################################################################
explosion_sounds = get_recursive_sound_names(EXPLOSION_SOUNDS, "./")
motor_sounds = get_recursive_sound_names(MOTOR_SOUNDS, "./")
wood_sounds = get_recursive_sound_names(WOOD_SOUNDS, "./")
human_sounds = get_recursive_sound_names(HUMAN_SOUNDS, "./")
nature_sounds = get_recursive_sound_names(NATURE_SOUNDS, "./")
domestic_sounds = get_recursive_sound_names(DOMESTIC_SOUNDS, "./")
tools = get_recursive_sound_names(TOOLS_SOUNDS, "./")
#wild_animals=get_recursive_sound_names(Wild_animals)
'''
pos_sounds = {}
neg_sounds = {}
for label_dicts in config["labels"]:
    lab_name = label_dicts["aggregatePositiveLabelName"]
    pos_temp = get_recursive_sound_names(label_dicts["positiveLabels"],"/home/madlad/Code_Projects/modular_acoustic_detection/")
    pos_sounds[lab_name] = pos_temp
    if label_dicts["negativeLabels"]!=None:
        neg_lab_name = label_dicts["aggregateNegativeLabelName"]
        neg_temp = get_recursive_sound_names(label_dicts["negativeLabels"],"/home/madlad/Code_Projects/modular_acoustic_detection/")
        neg_temp = neg_temp.difference(pos_temp)
        neg_sounds[neg_lab_name] = neg_temp


pprint(pos_sounds)
pprint(neg_sounds)


########################################################################
      # Importing balanced data from the function.
      # Including audiomoth annotated files for training
########################################################################
DATA_FRAME = balancing_dataset.balanced_data(audiomoth_flag=0, mixed_sounds_flag=0)

def subsample_dataframe(dataframe, subsample):
  """
  Subsample examples from the dataframe
  """
  if subsample is not None:
    if subsample > 0:
      # If subsample is less than size of dataframe, then
      # don't allow replacement when sampling
      # Otherwise, the intention is to oversample
      dataframe = dataframe.sample(subsample, replace=(subsample > dataframe.shape[0]))
    else:
      dataframe = pd.DataFrame([], columns=dataframe.columns)

  return dataframe

def split_and_subsample_dataframe(dataframe, validation_split, subsample):
  """
  Perform validation split and sub/over sampling of dataframe
  Returns train and test dataframe
  """

  # split before sub/oversampling to ensure there is no leakage between train and test sets
  test_size = int(validation_split*dataframe.shape[0])

  train_size = dataframe.shape[0] - test_size

  if test_size == 0:
    train_df = dataframe
    test_df = pd.DataFrame({}, columns=dataframe.columns)
  elif train_size == 0:
    train_df = pd.DataFrame({}, columns=dataframe.columns)
    test_df = dataframe
  else:
    train_df, test_df = train_test_split(dataframe,
                                         test_size=test_size,
                                         random_state=42)

  test_subsample = int(subsample*validation_split)
  train_subsample = subsample - test_subsample

  if (train_df.shape[0] == 0) and (train_subsample > 0):
    print(Fore.RED, "No examples to subsample from!", Style.RESET_ALL)
    #assert(False)
  else:
    train_df = subsample_dataframe(train_df, train_subsample)

  if (test_df.shape[0] == 0) and (test_subsample > 0):
    print(Fore.RED, "No examples to subsample from!", Style.RESET_ALL)
    #assert(False)
  else:
    test_df = subsample_dataframe(test_df, test_subsample)

  return train_df, test_df

def get_select_vector(dataframe, label_filter_arr):
  """
  Get the boolean select vector on the dataframe from the label filter
  """
  return dataframe['labels_name'].apply(lambda arr: np.any([x.lower() in label_filter_arr for x in arr]))

def import_dataframes(dataframe_file_list,
                      positive_label_filter_arr,
                      negative_label_filter_arr,
                      validation_split):
  """
  Iterate through each pickle file and import a subset of the dataframe
  """

  # Proceed with importing all dataframe files
  list_of_train_dataframes = []
  list_of_test_dataframes = []
  for input_file_dict in dataframe_file_list:

    assert("patternPath" not in list(input_file_dict.keys()))
    assert("path" in list(input_file_dict.keys()))

    print("Importing", input_file_dict["path"], "...")

    with open(input_file_dict["path"], 'rb') as file_obj:

      # Load the file
      df = pickle.load(file_obj)

      # Filtering the sounds that are exactly 10 seconds
      # Examples should be exactly 10 seconds. Anything else
      # is not a valid input to the model
      df = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]

      # Only use examples that have a label in label filter array
      positive_example_select_vector = get_select_vector(df, positive_label_filter_arr)
      positive_examples_df = df.loc[positive_example_select_vector]

      # This ensures there no overlap between positive and negative examples
      negative_example_select_vector = ~positive_example_select_vector
      if negative_label_filter_arr is not None:
        # Exclude even further examples that don't fall into the negative label filter
        negative_example_select_vector &= get_select_vector(df, negative_label_filter_arr)
      negative_examples_df = df.loc[negative_example_select_vector]

      # No longer need df after this point
      del df
      #if len(positive_examples_df)>0:
      train_positive_examples_df, test_positive_examples_df = \
                split_and_subsample_dataframe(dataframe=positive_examples_df,
                                              validation_split=validation_split,
                                              subsample=input_file_dict["subsample"])

      del positive_examples_df
      #if len(negative_examples_df)>0:
      train_negative_examples_df, test_negative_examples_df = \
                split_and_subsample_dataframe(dataframe=negative_examples_df,
                                              validation_split=validation_split,
                                              subsample=input_file_dict["subsample"])

      del negative_examples_df

      # append to overall list of examples
      #if len(train_negative_examples_df)>0:
      list_of_train_dataframes += [train_positive_examples_df, train_negative_examples_df]


      #if len(test_negative_examples_df)>0:
      list_of_test_dataframes += [test_positive_examples_df, test_negative_examples_df]
      
  train_df = pd.concat(list_of_train_dataframes, ignore_index=True)
  test_df = pd.concat(list_of_test_dataframes, ignore_index=True)

  print("Import done.")

  return train_df, test_df

final_dfs_train = []
final_dfs_test = []
for key in pos_sounds.keys():

    if key in neg_sounds.keys():
        DF_TRAIN, DF_TEST = \
            import_dataframes(dataframe_file_list=config["train"]["inputDataFrames"],
                              positive_label_filter_arr=pos_sounds[key],
                              negative_label_filter_arr=neg_sounds[key],
                              validation_split=config["train"]["validationSplit"])
    else:
        DF_TRAIN, DF_TEST = \
            import_dataframes(dataframe_file_list=config["train"]["inputDataFrames"],
                              positive_label_filter_arr=pos_sounds[key],
                              negative_label_filter_arr=None,
                              validation_split=config["train"]["validationSplit"])

    final_dfs_train.append(DF_TRAIN)
    final_dfs_test.append(DF_TEST)

df_train = pd.concat(final_dfs_train,ignore_index=True)
df_test = pd.concat(final_dfs_test,ignore_index=True)




########################################################################
      # Different classes of sounds.
      # You can increase the class by adding the necesssary sounds of that class
########################################################################
#ALL_SOUND_NAMES = ['Motor_sound', 'Explosion_sound', 'Human_sound',
#                   'Nature_sound', 'Domestic_animals', 'Tools']
#ALL_SOUND_LIST = list(explosion_sounds | motor_sounds | human_sounds | \
#                      nature_sounds | domestic_sounds | tools)


LABELS_BINARIZED_TRAIN = pd.DataFrame()
for key in pos_sounds.keys():
    FULL_NAME = key
    POSITIVE_LABELS = pos_sounds[key]
    LABELS_BINARIZED_TRAIN[FULL_NAME] = 1.0 * get_select_vector(df_train, POSITIVE_LABELS)

    LABELS_BINARIZED_TEST = pd.DataFrame()
    LABELS_BINARIZED_TEST[FULL_NAME] = 1.0 * get_select_vector(df_train, POSITIVE_LABELS)

print(LABELS_BINARIZED_TRAIN)

'''
########################################################################
      # Map all the sounds into their respective classes
      # Include the similar column if a new class label is to be added
########################################################################
DATA_FRAME['labels_new'] = DATA_FRAME['labels_name'].apply(lambda arr: ['Motor_sound' if x  in motor_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Explosion_sound' if x  in explosion_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Nature_sound' if x  in nature_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Human_sound' if x  in human_sounds else x for x in arr])
# DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Wood_sound' if x  in wood_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Domestic_animals' if x  in domestic_sounds else x for x in arr])
DATA_FRAME['labels_new'] = DATA_FRAME['labels_new'].apply(lambda arr: ['Tools' if x  in tools else x for x in arr])
# DATA_FRAME['labels_new']=DATA_FRAME['labels_new'].apply(lambda arr: ['Wild_animals' if x  in Wild_animals else x for x in arr])



########################################################################
      # Binarize the labels. Its a Multi-label binarizer
########################################################################
NAME_BIN = MultiLabelBinarizer().fit(DATA_FRAME['labels_new'])
LABELS_BINARIZED = NAME_BIN.transform(DATA_FRAME['labels_new'])
LABELS_BINARIZED_ALL = pd.DataFrame(LABELS_BINARIZED, columns=NAME_BIN.classes_)
LABELS_BINARIZED = LABELS_BINARIZED_ALL[ALL_SOUND_NAMES]


########################################################################
      # print out the number and percenatge of each class examples
########################################################################
print(LABELS_BINARIZED.mean())



########################################################################
      # Filtering the sounds that are exactly 10 seconds
########################################################################
DF_FILTERED = DATA_FRAME.loc[DATA_FRAME.features.apply(lambda x: x.shape[0] == 10)]
LABELS_FILTERED = LABELS_BINARIZED.loc[DF_FILTERED.index, :]
#print(LABELS_FILTERED)



########################################################################
      # split the data into train and test data
########################################################################
DF_TRAIN, DF_TEST, LABELS_BINARIZED_TRAIN, LABELS_BINARIZED_TEST = train_test_split(DF_FILTERED, LABELS_FILTERED,
                                                                                    test_size=0.33, random_state=42)



########################################################################
      # preprecess the data into required structure
########################################################################
X_TRAIN = np.array(DF_TRAIN.features.apply(lambda x: x.flatten()).tolist())
X_TRAIN_STANDARDIZED = X_TRAIN / 255
X_TEST = np.array(DF_TEST.features.apply(lambda x: x.flatten()).tolist())
X_TEST_STANDARDIZED = X_TEST / 255



########################################################################
      # create the keras model.
      # Change and play around with model architecture
      # Hyper parameters can be tweaked here
########################################################################
def create_keras_model():
    """
    Creating a Model
    """
    model = Sequential()
    model.add(Conv1D(500, input_shape=(1280, 1), kernel_size=128,
                     strides=128, activation='relu', padding='same'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    print(model.summary())
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4, epsilon=1e-8),
                  metrics=['accuracy'])
    return model




########################################################################
    # reshaping the train and test data so as to align with input for model
########################################################################
CLF2_TRAIN = X_TRAIN.reshape((-1, 1280, 1))
CLF2_TEST = X_TEST.reshape((-1, 1280, 1))
CLF2_TRAIN_TARGET = LABELS_BINARIZED_TRAIN
CLF2_TEST_TARGET = LABELS_BINARIZED_TEST



########################################################################
      # Implementing & Training the keras model
########################################################################
MODEL = create_keras_model()
MODEL_TRAINING = MODEL.fit(CLF2_TRAIN, CLF2_TRAIN_TARGET,
                           epochs=100, batch_size=500, verbose=1,
                           validation_data=(CLF2_TEST, CLF2_TEST_TARGET))




########################################################################
      # Predict on train and test data
      # Changing decision threshold can be done here
########################################################################
CLF2_TRAIN_PREDICTION = MODEL.predict(CLF2_TRAIN).round()
CLF2_TRAIN_PREDICTION_PROB = MODEL.predict(CLF2_TRAIN)
CLF2_TEST_PREDICTION = MODEL.predict(CLF2_TEST).round()
CLF2_TEST_PREDICTION_PROB = MODEL.predict(CLF2_TEST)




########################################################################
      # To get the Misclassified examples
########################################################################
DF_TEST['actual_labels'] = np.split(LABELS_BINARIZED_TEST.values, DF_TEST.shape[0])
DF_TEST['predicted_labels'] = np.split(CLF2_TEST_PREDICTION, DF_TEST.shape[0])
DF_TEST['predicted_prob'] = np.split(CLF2_TEST_PREDICTION_PROB, DF_TEST.shape[0])
MISCLASSIFED_ARRAY = CLF2_TEST_PREDICTION != CLF2_TEST_TARGET
MISCLASSIFIED_EXAMPLES = np.any(MISCLASSIFED_ARRAY, axis=1)




########################################################################
      # print misclassified number of examples
########################################################################
print('Misclassified number of examples :', DF_TEST[MISCLASSIFIED_EXAMPLES].shape[0])



########################################################################
      # Print confusion matrix and classification_report
########################################################################
print(CLF2_TEST_TARGET.values.argmax(axis=1).shape)
print('        Confusion Matrix          ')
print('============================================')
RESULT = confusion_matrix(CLF2_TEST_TARGET.values.argmax(axis=1),
                          CLF2_TEST_PREDICTION.argmax(axis=1))
print(RESULT)
print('        Classification Report      ')
print('============================================')
CL_REPORT = classification_report(CLF2_TEST_TARGET.values.argmax(axis=1),
                                  CLF2_TEST_PREDICTION.argmax(axis=1))
print(CL_REPORT)




########################################################################
        # calculate accuracy and hamming loss
########################################################################
ACCURACY = accuracy_score(CLF2_TEST_TARGET.values.argmax(axis=1),
                          CLF2_TEST_PREDICTION.argmax(axis=1))
HL = hamming_loss(CLF2_TEST_TARGET.values.argmax(axis=1), CLF2_TEST_PREDICTION.argmax(axis=1))
print('Hamming Loss :', HL)
print('Accuracy :', ACCURACY)




########################################################################
        # Save the model weights
        # Change the name if you are tweaking hyper parameters
########################################################################
MODEL.save_weights('multilabel_model_maxpool_version.h5')
'''