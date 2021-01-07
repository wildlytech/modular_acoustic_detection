"""
pre-processing the data
"""
# Import necessary libraries
import argparse
import pickle
import pandas as pd
from colorama import Fore, Style


###########################################################################
                # Description and help
###########################################################################

DESCRIPTION = "Reads the annotation file and Embeddings \
              from the selected folder and outputs the base dataframe"
HELP = "Input the annotation file path [Label_1, Label_2, Label_3] \n \
        and path for its embeddings"


###########################################################################
        #parse the input arguments given from command line
###########################################################################

PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-annotation_file', '--annotation_file', action='store',
                    help=HELP)
PARSER.add_argument('-path_for_saved_embeddings', '--path_for_saved_embeddings', action='store',
                    help="Input the path where embeddings are stored")
PARSER.add_argument('-path_to_save_dataframe', '--path_to_save_dataframe', action='store',
                    help="Input the path to save dataframe (.pkl) file")
RESULT = PARSER.parse_args()




LABELS_NAME_COLUMNS = ['Label_1', "Label_2", "Label_3", "Label_4"]


###########################################################################

###########################################################################
def read_data_files(filename):
    """
    read the annotated file
    (".csv")
    """
    data_files = pd.read_csv(filename).fillna("")
    return data_files


###########################################################################
                # create dictionary
###########################################################################
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



###########################################################################
            # Helper Function
###########################################################################
def check_for_null(array):
    """
    check for null string values in array
    """
    while "" in array:
        array.remove("") if "" in array else array
    return array


def preprocess_data(data_frame, label_columns_list, data_file_name):
    """
    start preprocessing the data
    """
    print('\npreprocessing..')
    print(Style.RESET_ALL)
    for col in label_columns_list:
        data_frame[col] = data_frame[col].apply(lambda arr: arr.strip(""))
        data_frame[col] = data_frame[col].replace(SET_DICTIONARY)
    data_frame['labels_name'] = data_frame[label_columns_list].values.tolist()
    data_frame['labels_name'] = data_frame['labels_name'].apply(lambda arr: check_for_null(arr))
    print(Fore.GREEN + "pre-processing Done:" + data_file_name.split("/")[-1])
    print(Style.RESET_ALL)
    # removing the null labelling rows
    index_null = data_frame['labels_name'].loc[data_frame['labels_name'].apply(lambda arr: len(arr) == 0)].index
    data_frame = data_frame.drop(index_null)
    data_frame.index = list(range(data_frame.shape[0]))
    return data_frame


def check_for_unknown_label(data_frame, label_columns_list):
    """
    """
    labels_not_found = []
    for col in label_columns_list:
        for each_label in data_frame[col].values.tolist():
            if each_label in list(SET_DICTIONARY.keys()) or each_label in list(SET_DICTIONARY.values()):
                pass
            else:
                labels_not_found.append(each_label)

    print("Labels not found in Dictionary: \n", list(set(labels_not_found)))



def read_embeddings(data_frame, path_to_embeddings):
    """
    read the embeddings
    """
    embeddings_list = []
    test_index = []
    for each_file in data_frame['wav_file'].values.tolist():
        try:
            with open(path_to_embeddings+each_file[:-4]+".pkl", 'rb') as file_obj:
                embeddings_list.append(pickle.load(file_obj))
        except:
            test_index.append(data_frame['wav_file'].values.tolist().index(each_file))
    data_frame = data_frame.drop(test_index)
    data_frame.index = list(range(data_frame.shape[0]))
    data_frame['features'] = embeddings_list
    return data_frame


def initiate_preprocessing(data_file_name, path_to_embeddings):
    """
    initiate preprocessing by
    reading data file
    """
    data_file = read_data_files(data_file_name)
    check_for_unknown_label(data_file, LABELS_NAME_COLUMNS)
    data = preprocess_data(data_file, LABELS_NAME_COLUMNS, data_file_name)
    data = data.drop(LABELS_NAME_COLUMNS, axis=1)

    #read all the embeddings
    if path_to_embeddings:
        data_with_embeddings = read_embeddings(data, path_to_embeddings)
        return data_with_embeddings
    else:
        return data


def write_dataframe(path_to_write, dataframe):
    """
    write out the dataframe in pickle format
    """
    if path_to_write:
        with open(path_to_write, "wb") as file_obj:
            pickle.dump(dataframe, file_obj)
    else:
        print("Input path to write dataframe")

###########################################################################
                # Main Function
###########################################################################
if __name__ == "__main__":

    if RESULT.path_for_saved_embeddings:
        DATAFRAME = initiate_preprocessing(RESULT.annotation_file, RESULT.path_for_saved_embeddings)
    else:
        DATAFRAME = initiate_preprocessing(RESULT.annotation_file, None)



    if RESULT.path_to_save_dataframe:
        write_dataframe(RESULT.path_to_save_dataframe, DATAFRAME)
    else:
        write_dataframe(None, DATAFRAME)
