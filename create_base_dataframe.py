import os
import pickle
import glob
import argparse
import pandas as pd

#########################################################################################
            # Helper Functions
#########################################################################################
def read_pickle_file(filename):
    """
    Reads the pickle file
    """
    with open(filename, "rb") as file_obj:
        pickle_value = pickle.load(file_obj)
    return pickle_value



def get_embeddings_path(path_for_saved_embeddings, audio_files_list):
    """
    Returns path of only existing embeddings file
    """
    found_embeddings_paths = []
    for each_value in audio_files_list:
        if os.path.exists(path_for_saved_embeddings+each_value[:-3]+"pkl"):
            found_embeddings_paths.append(path_for_saved_embeddings+each_value[:-3]+"pkl")
        else:
            pass
    print("No. Embeddings found as per given audio Filenames in dataframe: ", len(found_embeddings_paths))
    return found_embeddings_paths


def create_dataframe(column_names):
    """
    creates a dataframe with given column name
    """
    dataframe_inside_scope = pd.DataFrame(columns=column_names)
    return dataframe_inside_scope

def read_all_embeddings(embedding_filespath):
    """
    Reads all the embedding files in pickle format
    """
    embeddings_value = []
    for each_value in embedding_filespath:
        embeddings_value.append(read_pickle_file(each_value))

    return embeddings_value


def start_from_initial(embedding_filespath):
    """
    Reads all the embeddings and creates the dataframe
    """
    embeddings_value = []
    embeddings_name = []
    embeddings_names_with_path = glob.glob(embedding_filespath+"*.pkl")
    for each_value in embeddings_names_with_path:
        embeddings_value.append(read_pickle_file(each_value))
        embeddings_name.append(each_value.split("/")[-1][:-3]+"wav")
    dataframe_inside_scope = create_dataframe(["wav_file", "features"])
    dataframe_inside_scope["wav_file"] = embeddings_name
    dataframe_inside_scope["features"] = embeddings_value
    return dataframe_inside_scope


def write_dataframe(path_to_write, dataframe):
    """
    Write the dataframe in pickle format
    """
    with open(path_to_write, "wb") as file_obj:
        pickle.dump(dataframe, file_obj)


def replace_pkl_files_to_wavfiles(pklfiles_list):
    """
    Replaces all the pkl files into wav file extension
    """
    wavfiles_list = []
    for each_value in pklfiles_list:
        wavfiles_list.append(each_value.split("/")[-1][:-3]+"wav")
    return wavfiles_list


def create_new_dataframe(path_for_saved_embeddings, path_to_write_dataframe):
    """
    Creates a new dataframe from the embeddings
    """
    new_dataframe = start_from_initial(path_for_saved_embeddings)
    write_dataframe(path_to_write_dataframe, new_dataframe)

#########################################################################################
            # Main function
#########################################################################################
if __name__ == "__main__":

    #####################################################################################
                   # description and Help
    #####################################################################################

    DESCRIPTION = '1. Input the path for base dataframe [wav_file, labels_name] or \n \
                   2. Input the path for embeddings only, if base dataframe not present'
    HELP = 'Input the path'

    #####################################################################################
                # Arguments and parsing
    #####################################################################################
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    OPTIONAL_ARGUMENTS = PARSER.add_argument_group('optional arguments')

    REQUIRED_ARGUMENTS.add_argument('-path_for_saved_embeddings', '--path_for_saved_embeddings',
                                    action='store',
                                    help='Input the path for the folder where the embedding files (*.pkl) are saved',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-path_to_write_dataframe', '--path_to_write_dataframe',
                                    action='store',
                                    help='Input the path to write the dataframe (*.pkl)',
                                    required=True)
    OPTIONAL_ARGUMENTS.add_argument('-dataframe_without_feature', '--dataframe_without_feature',
                                    action='store',
                                    help='Input the path to existing dataframe (*.pkl) without features column')
    RESULT = PARSER.parse_args()

    if RESULT.dataframe_without_feature:
        assert(RESULT.dataframe_without_feature.endswith(".pkl"))
        ORIGINAL_DATAFRAME = read_pickle_file(RESULT.dataframe_without_feature)
        GET_LIST_EMBEDDINGS_PRESENT = get_embeddings_path(path_for_saved_embeddings=RESULT.path_for_saved_embeddings,
                                                          audio_files_list=ORIGINAL_DATAFRAME["wav_file"].tolist())
        FEATURES_LIST = read_all_embeddings(GET_LIST_EMBEDDINGS_PRESENT)
        NEW_DATAFRAME = create_dataframe(["wav_file", "features"])
        NEW_DATAFRAME["wav_file"] = replace_pkl_files_to_wavfiles(GET_LIST_EMBEDDINGS_PRESENT)
        NEW_DATAFRAME["features"] = FEATURES_LIST
        NEW_DATAFRAME = pd.merge(ORIGINAL_DATAFRAME, NEW_DATAFRAME, on="wav_file")
        write_dataframe(RESULT.path_to_write_dataframe, NEW_DATAFRAME)

    else:
        assert(RESULT.path_to_write_dataframe.endswith(".pkl"))
        create_new_dataframe(path_for_saved_embeddings=RESULT.path_for_saved_embeddings,
                             path_to_write_dataframe=RESULT.path_to_write_dataframe)

