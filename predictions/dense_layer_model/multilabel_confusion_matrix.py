"""
--version of python
  python 3.7
"""

import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix



##############################################################################
          # Description and Help
##############################################################################
DESCRIPTION = 'Prints the confusion matrix for each label separately \
			   for a multilabel inputs and targets.'



##############################################################################
          # Parsing the inputs given
##############################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-path_for_dataframe_with_inputs_targets', '--path_for_dataframe_with_inputs_targets',
                    action='store', help='Input the path for dataframe(.pkl format)')
RESULT = PARSER.parse_args()




##############################################################################
			# Reads the pickle file
##############################################################################
def read_pickle(path):
    """
    Read the pickle file
    """
    with open(path, "rb") as file_obj:
        dataframe = pickle.load(file_obj, encoding="latin1")

    return dataframe



##############################################################################
			# Reads the pickle file
##############################################################################
DATAFRAME = read_pickle(RESULT.path_for_dataframe_with_inputs_targets)





##############################################################################
			# read the ground truth values and predicted values
##############################################################################
GROUND_TRUTH_VALUES = np.array(DATAFRAME['actual_labels'].values.tolist())
PREDICTED_VALUES = np.array(DATAFRAME['predicted_labels'].values.tolist())




##############################################################################
		# Prints confusion matrix for each label separately
##############################################################################
print(multilabel_confusion_matrix(GROUND_TRUTH_VALUES, PREDICTED_VALUES))
