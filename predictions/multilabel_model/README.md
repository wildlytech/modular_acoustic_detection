# Predictions using Multilabel Models

<br>

## Predictions on Dataframes
- Generate the embeddings for each of ```.wav``` files and create a base dataframe with columns ```["wav_file", "labels_name", "features"]```
- Save the dataframe as a pickle file and add the path to the pickle file to the adequate key in the ```model_configs/multilabel_model/multilabel_maxpool.json```
- Run the ```predictions/multilabel_model/multilabel_pred.py``` file using the following arguments

### About the arguments
- ```-predictions_cfg_json``` : Path to the predictions config file
- ```-path_for_dataframe_with_features``` : Path to the dataframe pickle you want to predict on. Optional since input features can be a part of the predictions config
- ```-save_misclassified_examples``` : Path to the file where we want to save a csv of misclassified examples
- ```-path_to_save_prediction_csv```: Path to the file were the predictions file is to be saved

## Predictions on Audio Files
- Run the ```predictions/multilabel_model/multilabel_pred_on_wavfile.py``` along with the arguments listed below to make predictions on a single audio file
### About the Arguments
- ```-predictions_cfg_json```: Path to the predictions config file
- ```-path_for_wavfile```: Path to the wav file
 