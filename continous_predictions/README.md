# Predicting Local Folder files & FTP Folder files

<br>

## 1. Local Folder Files Predictions

- Use this script to predict ```.wav``` Files from any folder in local drive.
- This will keep looping over the files and predicts each file at a time and writes the prediction result of that file into ```csv``` row.
- You can keep refreshing ```.csv``` file to see the results as and when predictions are written to it.
- Follow the command to get the predictions ```.csv```  for ```.wav``` files in any local directory


```shell
$ python -m continous_predictions.batch_test_offline    -local_folder_path
                                                        -csv_filename
```

<br>

## 2. FTP Folder Files Predictions

- Use this script to predict ```.wav``` Files from any folder in FTP server.
- This will keep looping for the files and also wait for files to upload (```2 Minutes``` as default but can be changed in script), predicts each file at a time and writes the prediction result of that file into ```csv``` row.
- You can keep refreshing ```.csv``` file to see the results as and when predictions are written to it.
- Follow the command to get the predictions ```.csv```  for ```.wav``` files in any FTP directory

```shell
$ python -m continous_predictions.batch_test_ftp_files  -ftp_folder_path
                                                        -csv_filename
```
###### Note
- FTP credentials must be passed in script [here](https://github.com/wildlytech/modular_acoustic_detection/blob/3a0b30c5f8590ea6130eb0cfc5ec35132172f318/continous_predictions/batch_test_ftp_files.py#L26)
```python
FTP_PASSWORD = "********"
```
- You can change the wait time from default 2 Minutes ```(120 seconds)``` to any required duration in ```seconds``` [here](https://github.com/wildlytech/modular_acoustic_detection/blob/3a0b30c5f8590ea6130eb0cfc5ec35132172f318/continous_predictions/batch_test_ftp_files.py#L230)
```python
time.sleep(120)
```

