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
$ python binary_model.py
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
$ python multilabel_dense_model.py
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
$ python multilabel_maxpool_model.py
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
- Toggle the flag in below snippet of  [binary_relevance_model.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/models/binary_relevance_model.py) if we need include audiomoth sounds, mixed sounds as well. By default only Google audioset is used.
```python
DATA_FRAME = balancing_dataset.balanced_data(audiomoth_flag=0, mixed_sounds_flag=0)
```
- Change column names as per the label's that are to be detected and start the training process. Below the snippet that needs to changed to for different Binary Relevance Models. 
- Example shown is Domestic Binary Relevance Model training. [Line 79: binary_relevance_model.py](https://github.com/wildlytech/modular_acoustic_detection/blob/28a38658a659ddabbd4d73cfad3c91132ab3736e/models/binary_relevance_model.py#L79)
```python
LABELS_BINARIZED["Domestic"] = NEW_LABELS_BINARIZED['Domestic_animals']
```
- Change also the weight file name as per the model in [Line 223: binary_relevance_model.py](https://github.com/wildlytech/modular_acoustic_detection/blob/28a38658a659ddabbd4d73cfad3c91132ab3736e/models/binary_relevance_model.py#L223)
```python
MODEL.save_weights('binary_relevance_domestic_model.h5')
```
- To train a Binary Relevance model follow the command shown below
```shell
$ python binary_relevance_model.py
```



