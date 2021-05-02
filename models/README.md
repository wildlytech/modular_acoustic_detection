# Training ML / DL  Models

<br>

***

#### 1. Multi-Label Model
This a Multilabel Model version, where in we have X(Features i.e 128x10 vector) and Target as Y(Label i.e labels_name) which is now grouped into multiple (7) Broader categories which against contains sub labels in it.
-   It’s called Maxpool layer model because prediction layer(Final Layer) is a Maxpool layer.
- Prerequisite : Balancing the data is the prerequisite. It is different from the balancing procedure followed as in case of Binary label model as here there are multi-labels.

- To train a multi-label model follow the command shown below
```shell
$ python -m models.multilabel_model [-h] -cfg_json MODEL_CONFIG_JSON_FILE
```

<br>

***

#### 2. Binary Relevance Models
-   Training here involves creating a separate model for each label. Hence as many labels to be detected so many binary relevance models should be created or saved
-  Prerequisite :  Balancing of data in Binary Relevance Models. It’s a one v/s all kind of Balancing method.
-   For example the model to be trained is a Motor Binary Relevance Model, then ```Impact sound(1’s)``` will be Motor and rest other sounds are ```Ambient (0’s)``` i.e [Explosion, Human, Nature, Domestic and Tools ] and if model to be trained is an Explosion Binary Relevance Model then Impact sounds(1’s) will be Explosion and rest other sounds are Ambient(0’s) i.e [Motor, Human, Nature, Domestic and Tools]
- Balancing can be achieved within the model configuration json file, which is a parameter to train the binary relevance model.  Under ```["train"]["inputDataFrames"]``` lies a list of all the dataframes with sound embedding features and labels. The ```subsample``` parameter allows for taking only a subset or upsampled set of the examples in the pickle file. Use this parameter to balance data from different sources.
- To train a Binary Relevance model follow the command shown below
```shell
$ python -m models.binary_relevance_model [-h]  -model_cfg_json MODEL_CONFIG_JSON_FILE
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
    - Initial PANNs weights (e.g. `final_5fold_sed_dense121_nomix_fold0_checkpoint_50_score0.7219`)
    - Folder containing split wavfiles (10 seconds) of corresponding audioclips
- If using custom data make sure to bring your data files to these formats

- To run Panns finetune: run `python models/panns_finetune.py` with the 
appropriate arguments from the help menu.
- Make sure to adjust the relevant filepaths and parameters in the `model_configs/panns/train_config.json`
