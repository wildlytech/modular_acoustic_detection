# Predictions using Maxpool Layer Model

<br>

## 1. When Data is Labelled
- This a Multilabel Model version, where in we have X(Features i.e 128*10 vector) and Target as Y(Label i.e labels_name) as it annotated which is  grouped into 6 Broader categories which again contains sub labels in it.
-   It’s called Maxpool layer model because prediction layer(Final Layer) is a Maxpool layer
- Generate the embeddings for each of ```.wav``` files and create a base dataframe of ```TYPE 1``` with columns ```["wav_file", "labels_name", "features"]```
- Input the whole ```TYPE 1``` dataframe to get the predictions
- Follow the command below to get the ```precision```, ```recall```, ```confusion matrix```, ```classification report``` for the data
-   When data in the pickle dataframe format (Type 1) then testing involves logging some of the metrics and evaluating the model performance.
- About Arguments : [get_results_multilabel_maxpool.py](https://github.com/wildlytech/modular_acoustic_detection/blob/b9ecc45b698a7be8a266fc463f4489cdff3d688e/predictions/maxpool_layer_model/get_results_multilabel_maxpool.py#L37)
    - ```-path_for_dataframe_with_FL``` : Path for ```TYPE 1``` dataframe file with ```(.pkl)```  extension
    - ```-path_for_maxpool_saved_weights_file``` : Path where model weights are saved ```.h5``` file
    - ```-csvfilename_to_save_predictions``` : Path for csv file to save all the prediction results ```(.csv)``` file
    - ```-path_to_save_prediciton_dataframe``` : Path for ```.pkl``` file to save all the prediction results in dataframe format

```shell
$ python -m predictions.maxpool_layer_model.get_results_multilabel_maxpool [-h] -path_for_dataframe_with_FL
                                                                                -path_for_dense_saved_weights_file
                                                                                -csvfilename_to_save_predictions
                                                                                -path_to_save_prediciton_dataframe
```

<br>

## 2. When Data is Unlabeled

- This a Multilabel Model version, where in we have X(Features i.e 128*10 vector) and no Target Y (labels) as these files are not annotated
-   It’s called Maxpool layer model because prediction layer(Final Layer) is a Maxpool layer
- Generate the embeddings for each of ```.wav``` files and create a base dataframe of ```TYPE 2``` with columns ```["wav_file", "features"]```
- Follow the command below to get the predictions for files.
- **Note** : We won't be able to get ```precision```, ```recall```, ```confusion matrix```, ```classification report``` for the data as they are not annotated i.e no ground truth values are present for it
 - About Arguments : [predict_on_dataframe_file_maxpool.py](https://github.com/wildlytech/modular_acoustic_detection/blob/3a05ea41746ba72212f8878b519696b6d520258f/predictions/dense_layer_model/predict_on_dataframe_file.py#L25)
    - ```-path_for_dataframe_with_features``` : Path for ```TYPE 2``` dataframe file with ```(.pkl)```  extension
    - ```-path_for_maxpool_model_weights``` : Path where model weights are saved ```.h5``` file
    - ```-csvfilename_to_save_predictions``` : Path for csv file to save all the prediction results ```(.csv)``` file

```shell
python -m predictions.maxpool_layer_model.predict_on_dataframe_file [-h]    -path_for_dataframe_with_features
                                                                            -path_for_maxpool_model_weights
                                                                            -csvfilename_to_save_predictions
```



