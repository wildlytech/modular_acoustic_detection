"""
Generates the csv file with prediction results
"""
import glob
import argparse
from .batch_test_ftp_files import start_batch_run

############################################################################
# Loops over the list of files in the directory specified
############################################################################
def start_batch_run_offline(predictions_cfg_json, path_for_folder, csv_filename):
    """
    Writes the predicted results on to csvfile row wise
    """
    all_wav_files_path = glob.glob(path_for_folder + "*.WAV") + glob.glob(path_for_folder + "*.wav")
    all_wav_files = [each_file.split("/")[-1] for each_file in all_wav_files_path]
    print('Total No. of Files: ', len(all_wav_files))

    start_batch_run(predictions_cfg_json=predictions_cfg_json,
                    path_for_folder=path_for_folder,
                    wav_files_list=all_wav_files,
                    csv_filename=csv_filename,
                    online=False)

############################################################################
            # Main Function
############################################################################
if __name__ == "__main__":

    ########################################################################
    # Description and Help
    ########################################################################
    DESCRIPTION = "Generates the csv file with prediction results"
    HELP = "Give the Required Arguments"


    ########################################################################
    # Parsing argument
    ########################################################################
    ARGUMENT_PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    OPTIONAL_NAMED = ARGUMENT_PARSER._action_groups.pop()

    REQUIRED_NAMED = ARGUMENT_PARSER.add_argument_group('required arguments')
    REQUIRED_NAMED.add_argument('-local_folder_path', '--local_folder_path', action='store',
                        help='Input the path', required=True)
    REQUIRED_NAMED.add_argument('-predictions_cfg_json',
                        '--predictions_cfg_json',
                        help='Input (binary relevance) json configuration file for predictions output',
                        required=True)
    OPTIONAL_NAMED.add_argument('-csv_filename', '--csv_filename', action='store',
                        help='Input the name of csv file to save results', default='offline_predictions.csv')
    ARGUMENT_PARSER._action_groups.append(OPTIONAL_NAMED)
    RESULT = ARGUMENT_PARSER.parse_args()

    ########################################################################
    # Setting the input arguments
    ########################################################################
    FOLDER_FILES_PATH = RESULT.local_folder_path
    CSV_FILENAME = RESULT.csv_filename
    PREDICTIONS_CFG_JSON = RESULT.predictions_cfg_json

    start_batch_run_offline(PREDICTIONS_CFG_JSON, FOLDER_FILES_PATH, CSV_FILENAME)
