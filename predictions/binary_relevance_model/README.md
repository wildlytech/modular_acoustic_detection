# Predictions using Binary Relevance Models

<br>

## 1. When Data is Labelled
-  Generate the embeddings for each of ```.wav``` files and create a base dataframe of ```TYPE 1```  [as the data is annotated] with columns ```["wav_file", "labels_name", "features"]```
- Input the whole ```TYPE 1``` dataframe to get the predictions
- Follow the command below to get the ```precision```, ```recall```, ```confusion matrix```, ```classification report``` for the data
-   When data in the pickle dataframe format (Type 1) then testing involves logging some of the metrics and evaluating the model performance.
- About Arguments : [get_results_binary_relevance.py](https://github.com/wildlytech/modular_acoustic_detection/blob/3a05ea41746ba72212f8878b519696b6d520258f/predictions/dense_layer_model/get_results_multilabel.py#L37)
	- ```-path_for_dataframe_with_features``` : Path for ```TYPE 1``` dataframe file with ```(.pkl)```  extension
	- ```-save_misclassified_examples``` : File Path to save misclassified examples in ```.pkl``` 
	- ```-path_to_save_prediction_csv``` : Path for ```.csv``` file to save all the prediction results ```(.csv)``` file

```shell
$ python get_results_binary_relevance.py  [-h]   -path_for_dataframe_with_features
                                                 -save_misclassified_examples
                                                 -path_to_save_prediction_csv
```

<br>

## 2. When Data is Unlabeled

- Generate the embeddings for each of ```.wav``` files and create a base dataframe of ```TYPE 2``` with columns ```["wav_file", "features"]```
- Follow the command below to get the predictions for files.
- **Note** : We won't be able to get ```precision```, ```recall```, ```confusion matrix```, ```classification report``` for the data as they are not annotated i.e no ground truth values are present for it
 - About Arguments : [get_predictions_on_dataframe.py](https://github.com/wildlytech/modular_acoustic_detection/blob/3a05ea41746ba72212f8878b519696b6d520258f/predictions/dense_layer_model/predict_on_dataframe_file.py#L25)
	- ```-path_for_dataframe_with_features``` : Path for ```TYPE 2``` dataframe file with ```(.pkl)```  extension
	- ```-results_in_csv``` : Path for csv file to save all the prediction results ```(.csv)``` file

```shell
$ python get_predictions_on_dataframe.py  [-h]  -path_for_dataframe_with_features
                                                -results_in_csv
```



