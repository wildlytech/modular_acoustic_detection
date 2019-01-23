"""
pre-processing the data
"""
# Import necessary libraries
import pickle
import pandas as pd
import numpy as np
from colorama import Fore, Style

def read_data_files(filename):
    """
    read the annotated file
    (".csv")
    """
    # Read a data file
    data_files = pd.read_csv(filename).fillna("")
    return data_files

# create dictionary
SET_DICTIONARY = {"crow":"Crow",
                  "honking":"Vehicle",
                  "stream":"Stream",
                  "frogmouth":"Frog",
                  "birdf":"Bird",
                  "conersation":"Conversation",
                  "honkiong":"Vehicle",
                  "peafowl":"Bird",
                  "convertsation":"Conversation",
                  "inesct":"Insect",
                  "helicopter":"Vehicle",
                  "aeroplane":"Vehicle",
                  "plane":"Vehicle",
                  "birtd":"Bird",
                  "frog":"Frog",
                  "raini":"Rain",
                  "rain":"Rain",
                  "forg":"Frog",
                  "insect":"Insect",
                  "manmade":"Conversation",
                  "thunder":"Thunder",
                  "honkinig":"Vehicle",
                  "conversatoin":"Conversation",
                  "none":"",
                  "vehicle":"Vehicle",
                  "music":"Music",
                  "dog barking":"Dog",
                  "human":"Speech",
                  "conservation":"Conversation",
                  "conversation":"Conversation",
                  "bird":"Bird",
                  "felling (axe)":"Tools",
                  "wind":"Wind",
                  "biird":"Bird",
                  "footsteps":"Walk, footsteps",
                  "door closing":"",
                  "buzzing":"Insect",
                  "Silence" :"Silence",
                  "twig snap":"",
                  "buzz":"Insect",
                  "fly/buzzing":"Insect",
                  "----":''}

def check_for_null(array):
    """
    check for null string values in array
    """
    while "" in array:
        array.remove("") if "" in array else array
    return array


def preprocess_data(data_frame, label_columns_list,data_file_name):
    """
    start preprocessing the data
    """
    print '\npreprocessing..'
    print Style.RESET_ALL
    for col in label_columns_list:
        data_frame[col] = data_frame[col].apply(lambda arr: arr.strip(""))
        data_frame[col] = data_frame[col].replace(SET_DICTIONARY)
    data_frame['labels_name'] = data_frame[label_columns_list].values.tolist()
    data_frame['labels_name'] = data_frame['labels_name'].apply(lambda arr: check_for_null(arr))
    print Fore.GREEN + "pre-processing Done:" + data_file_name.split("/")[-1]
    print Style.RESET_ALL
    # removing the null labelling rows
    index_null = data_frame['labels_name'].loc[data_frame['labels_name'].apply(lambda arr: len(arr) == 0)].index
    data_frame = data_frame.drop(index_null)
    data_frame.index = range(data_frame.shape[0])
    return data_frame


def read_embeddings(data_frame, path_to_embeddings):
    """
    read the embeddings
    """
    embeddings_list = []
    test_index = []
    for each_file in data_frame['wav_file'].values.tolist():
        try:
            with open(path_to_embeddings+each_file+".pkl", 'rb') as file_obj:
                embeddings_list.append(pickle.load(file_obj))
        except:
            test_index.append(data_frame['wav_file'].values.tolist().index(each_file))
    data_frame = data_frame.drop(test_index)
    data_frame.index = range(data_frame.shape[0])
    data_frame['features'] = embeddings_list
    return data_frame

def initiate_preprocessing(data_file_name, path_to_embeddings):
    """initiate preprocessing by
    reading data file
    """
    data_file = read_data_files(data_file_name)
    data = preprocess_data(data_file, ['Label_1', "Label_2", "Label_3"], data_file_name)
    data = data.drop(['Label_1', "Label_2", "Label_3", "Location"], axis=1)

    #read all the embeddings
    data_with_embeddings = read_embeddings(data, path_to_embeddings)
    return data_with_embeddings
