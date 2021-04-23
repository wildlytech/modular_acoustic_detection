# Predictions using trained ML/DL models

<br>

#### 1. Multilabel Model
- Navigate to ```multilabel_model/``` folder to start predicting on single/multiple audio files using Multilabel Model
- To navigate follow the command below
```shell
$ cd dense_layer_model/
```
**Note**: [Navigate to multilabel_model/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/predictions/multilabel_model)

<br>

#### 2. Binary Relevance Models
- Navigate to ```binary_relevance_model/``` folder to start predictng on single/multiple audio files using different Binary Relevance models i.e each label will have a separate Trained Binary Relevance Model
- To navigate follow the command below
```shell
$ cd binary_relevance_model/
```
**Note**: [Navigate to binary_relevance_model/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/predictions/binary_relevance_model)

####4. Panns Inference on birds

- Make sure you are using this model to make inference on birds data.
- Make sure you have all the files moved to the repository as mentioned in 
the models/README.md under PANNs model.
- Run the panns inference script using the command `python predictions/panns/panns_infer_birds.py` with the
appropriate arguments mentioned in the help menu

####5. Panns Inference on audioset

- Make sure you are only using the script to make inference on audioset data
- Run the panns inference script using the command `python predictions/panns/panns_infer_audioset.py` with the
appropriate arguments mentioned in the help menu.
