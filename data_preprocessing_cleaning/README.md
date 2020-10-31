# Data Pre-process and Cleaning

<br>

#### 1. Separating Sounds Based on Labels
- This script will read the dataframe ```(TYPE 1)``` i.e it should have ```["labels_name"]``` column in it ,  separates out the sounds based on labeling, creates a different dataframe as per labels and writes at target path given.
- You can check [coarse_labels.csv](https://github.com/wildlytech/modular_acoustic_detection/blob/master/data_preprocessing_cleaning/coarse_labels.csv) file to know how  mapping of the labels and the separation of each sounds takes place.
- To separate sounds based on labels navigate to [separating_different_sounds.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/data_preprocessing_cleaning/seperating_different_sounds.py)
- Follow the command below to execute the script
- About Arguments
    - ```-dataframe_path``` : Path of the dataframe with ```.pkl``` extension. It should be ```TYPE 1``` dataframe i.e it should have ```["labels_name"]``` column in it
    - ```-path_to_write_different_sounds``` : Directory path to write the separated dataframes based on labels

```shell
$ python -m data_preprocessing_cleaning.seperating_different_sounds [-h]    -dataframe_path
                                                                            -path_to_write_different_sounds
```

<br>

***
#### 2. Splitting wav file into Chunks of Files
- Using this script we will be able to break down a single audio file (Usually of large audio length) into  number of chunks (Usually of ```10seconds``` each) of smaller audio length.
- About Argument:
    - ```-target_audio_file``` : Path of an audio wav file with ```.wav``` extension (Should be greater than ```10seconds``` which is default)
    - ```-path_to_write_chunks``` : Directory path to write the chunks of audio files (```10seconds``` each)

```shell
$ python -m data_preprocessing_cleaning.split_wav_file [-h] -target_audio_file
                                                            -path_to_write_chunks
```


**Note** : If we need to change default ```10seconds``` chunk to custom value. Change the following ```10000```  (in milliseconds) to ```custom value``` (in milliseconds) in snippet of [Line 86: split_wav_file.py](https://github.com/wildlytech/modular_acoustic_detection/blob/74844189e9fd12b7200c6d7dca47cda740d7e712/data_preprocessing_cleaning/split_wav_file.py#L86)
```python
if __name__ == "__main__":
    initiate_audio_split(RESULT.target_audio_file, 10000)
```

<br>

***
#### 3. Identifying Mislabeled silence sounds
- It identifies sounds that are actually silent (i.e its ```dB levels ``` are lowest ```-inf``` )  but labelled wrongly as something else and Vice-versa i.e sounds which are labelled as silence but audio file has some sound in it
- About Arguments:
    - ```-dataframe_path``` : Path for dataframe file with ```.pkl``` extension of TYPE 1 i.e it should have ```["labels_name"]``` column in it.
    - ```-path_for_audio_files``` : Directory path of the audio files (```.wav``` files)

```shell
$ python -m data_preprocessing_cleaning.identifying_mislabelled_silence_audiofiles [-h] -dataframe_path
                                                                                        -path_for_audio_files
```



