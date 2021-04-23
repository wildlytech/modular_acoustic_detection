# Training ML / DL  Models
<br>


#### 1. Binary Model

- Logistic Regression and Keras Binary Model
- We use these Binary model to train ML (Logistic Regression) model and DL (Keras Model) model to be able to detect if there is any ```Impact sounds``` i.e ```1's``` or ```Ambient sounds``` i.e```0's``` in an audio file
-  Check for what Impact and Ambient sounds consists of in [youtube_audioset.py](https://github.com/wildlytech/modular_acoustic_detection/blob/28a38658a659ddabbd4d73cfad3c91132ab3736e/youtube_audioset.py#L159)
-  Prerequisite : Balancing the data is the prerequisite which can be made in [balancing_dataset.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/balancing_dataset.py) script
- Toggle the flag in below snippet of  [binary_model.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/models/binary_model.py) if we need include audiomoth sounds, mixed sounds as well. By default only Google audioset is used.
```python
DATA_FRAME = balancing_dataset.balanced_data(audiomoth_flag=0, mixed_sounds_flag=0)
```
- To train a simple binary model follow the command shown below
```shell
$ python -m models.binary_model
```


<br>

***

#### 2. Multi-Label Dense Layer Model
-   This a Multilabel Model version, where in we have X(Features i.e 128*10 vector) and Target as Y(Label i.e labels_name) which is now grouped into 6 Broader categories which against contains sub labels in it.
-   It’s called Dense layer model because prediction layer(Final Layer) is a dense layer (Fully connected layer)
- Prerequisite : Balancing the data is the prerequisite which can be made in [balancing_dataset.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/balancing_dataset.py) script. It is different from the balancing procedure followed as in case of Binary label model as here there are multi-labels
- Toggle the flag in below snippet of  [multilabel_dense_model.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/models/multilabel_dense_model.py) if we need include audiomoth sounds, mixed sounds as well. By default only Google audioset is used.
```python
DATA_FRAME = balancing_dataset.balanced_data(audiomoth_flag=0, mixed_sounds_flag=0)
```
- To train a Dense Layer mutli-label model follow the command shown below
```shell
$ python -m models.multilabel_dense_model
```
- Change the weight file name as per your trials [Line 228: multilabel_dense_model.py](https://github.com/wildlytech/modular_acoustic_detection/blob/28a38658a659ddabbd4d73cfad3c91132ab3736e/models/multilabel_dense_model.py#L228)
```python
MODEL.save_weights('multiclass_weights.h5')
```

<br>

***

#### 3. Multi-Label Maxpool Layer Model
This a Multilabel Model version, where in we have X(Features i.e 128*10 vector) and Target as Y(Label i.e labels_name) which is now grouped into 6 Broader categories which against contains sub labels in it.
-   It’s called Maxpool layer model because prediction layer(Final Layer) is a Maxpool layer.
- Prerequisite : Balancing the data is the prerequisite which can be made in [balancing_dataset.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/balancing_dataset.py) script. It is different from the balancing procedure followed as in case of Binary label model as here there are multi-labels, but it same as in case of Dense layer Model
- Toggle the flag in below snippet of  [multilabel_maxpool_model.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/models/multilabel_maxpool_model.py) if we need include audiomoth sounds, mixed sounds as well. By default only Google audioset is used.
```python
DATA_FRAME = balancing_dataset.balanced_data(audiomoth_flag=0, mixed_sounds_flag=0)
```
- To train a Dense Layer mutli-label model follow the command shown below
```shell
$ python -m models.multilabel_maxpool_model
```
- Change the weight file name as per your trials [Line 223: multilabel_maxpool_model.py](https://github.com/wildlytech/modular_acoustic_detection/blob/28a38658a659ddabbd4d73cfad3c91132ab3736e/models/multilabel_maxpool_model.py#L228)
```python
MODEL.save_weights('multilabel_model_maxpool_version.h5')
```

<br>

***

#### 4. Binary Relevance Models
-   Training here involves creating a separate model for each label. Hence as many labels to be detected so many binary relevance models should be created or saved
-  Prerequisite :  Balancing of data in Binary Relevance Models. It’s a one v/s all kind of Balancing method.
-   For example the model to be trained is a Motor Binary Relevance Model, then ```Impact sound(1’s)``` will be Motor and rest other sounds are ```Ambient (0’s)``` i.e [Explosion, Human, Nature, Domestic and Tools ] and if model to be trained is an Explosion Binary Relevance Model then Impact sounds(1’s) will be Explosion and rest other sounds are Ambient(0’s) i.e [Motor, Human, Nature, Domestic and Tools]
- Balancing can be achieved within the model configuration json file, which is a parameter to train the binary relevance model.  Under ```["train"]["inputDataFrames"]``` lies a list of all the dataframes with sound embedding features and labels. The ```subsample``` parameter allows for taking only a subset or upsampled set of the examples in the pickle file. Use this parameter to balance data from different sources.
```
- To train a Binary Relevance model follow the command shown below
```shell
$ python -m models.binary_relevance_model [-h]  -model_cfg_json MODEL_CONFIG_JSON_FILE
                                                [-output_weight_file OUTPUT_WEIGHT_FILE]
```

#### 5. Parameter Search file
- This file takes parameters from the ```params_file.json``` file as input and prints out results after training the model using each of them.
- A variety of hyperparameters can be tested together 
- The hyperparameters supported include:<br>
a. loss: The loss function<br>
b. lr: The learning rate<br>
c. epsilon: Momentum parameter<br>
d. batch_size: Batch size<br>
e. networkCfgJson: Network architecture json file<br>
- The script can be run using the following command:
```python -m models.param_search -params_path <Path to the parameter json file> -config_path <Path to the model config file>```


#### 6. Panns Finetuning File

- The file finetunes the Panns model based on input data
- Make sure to run it only if you have a GPU Machine
- Additional files required to for testing
    - `checked_df_train.csv`: Csv of training audio clip names
    - `checked_df_test.csv` : Csv of testing audio clip names
    - `final_5fold_sed_dense121_nomix_fold0_checkpoint_50_score0.7219`: PANNs weights
    - Folder containing split wavfiles (10 seconds) of corresponding audioclips
- If using custom data make sure to bring your data files to these formats

- To run Panns finetune: run `python models/panns_xeno_finetune.py` with the 
appropriate arguments from the help menu.
- Make sure to adjust the relevant filepaths and parameters in the `model_configs/panns/train_config.py`
