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

<br>

#### 4. Converting audio clips in folder to dataframe
- From folder of audio clips, generate dataframe with labeled examples that can be used for training/evaluation.
- For xenocanto audio, use the xenocanto_to_dataframe.py script, which also does scraping in addition to transformations done in this script.
```shell
$ python -m data_preprocessing_cleaning.audio_clips_to_dataframe [-h] -f FOLDER_PATH
                                                                 [--delete_mp3_files]
                                                                 [--dont_delete_mp3_files]
                                                                 [--delete_wav_files]
                                                                 [--dont_delete_wav_files]
                                                                 [--delete_split_wav_files]
                                                                 [--dont_delete_split_wav_files]
                                                                 [--delete_embeddings_files]
                                                                 [--dont_delete_embeddings_files]
```

<br>

#### 5. Add labels to dataframe
- Add a labels_name column to dataframe pickle file from specified columns in csv file.
- If dealing with a xenocanto dataframe, the csv file is usually in a different format, so use add_xenocanto_labels_to_dataframe.py instead.
- To quickly get a csv file to label and add as an argument to this script, one can use apply pickle_to_csv.py on the dataframe to obtain a record of the audio clips in the dataframe

```shell
$ python -m data_preprocessing_cleaning.add_csv_labels_to_dataframe [-h] -d DATAFRAME
                                                                    -csv CSV
                                                                    -cols COLUMN_NAMES [COLUMN_NAMES ...]
                                                                    -dj DATAFRAME_JOIN_COLUMN
                                                                    -cj CSV_JOIN_COLUMN
                                                                    [-a] [-o OUTPUT_FILE_NAME]
```

<br>

#### 6. Remove rows from dataframe
- Remove rows from dataframe pickle file using entries in blacklist file.
```shell
$ python -m data_preprocessing_cleaning.remove_rows_from_dataframe.py [-h] -d DATAFRAME
                                                                      -b BLACKLIST_FILE
                                                                      -c ID_COLUMN
                                                                      [-s BLACKLIST_ITEM_SUFFIX]
                                                                      [-o OUTPUT_FILE_NAME]
```

<br>
