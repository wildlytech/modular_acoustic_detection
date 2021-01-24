# Predicting Local Folder files & FTP Folder files

<br>

## 1. Local Folder Files Predictions

- Use this script to predict ```.wav``` Files from any folder in local drive.
- This will keep looping over the files and predicts each file at a time and writes the prediction result of that file into ```csv``` row.
- You can keep refreshing ```.csv``` file to see the results as and when predictions are written to it.
- Follow the command to get the predictions ```.csv```  for ```.wav``` files in any local directory


```shell
$ python -m continous_predictions.batch_test_offline [-h] -local_folder_path LOCAL_FOLDER_PATH
                                                          -predictions_cfg_json PREDICTIONS_CFG_JSON
                                                          [-csv_filename CSV_FILENAME]
```

<br>

## 2. FTP Folder Files Predictions

- Use this script to predict ```.wav``` Files from any folder in FTP server.
- This will keep looping for the files and also wait for files to upload (```2 Minutes``` as default but can be changed in script), predicts each file at a time and writes the prediction result of that file into ```csv``` row.
- You can keep refreshing ```.csv``` file to see the results as and when predictions are written to it.
- Follow the command to get the predictions ```.csv```  for ```.wav``` files in any FTP directory

```shell
$ python -m continous_predictions.batch_test_ftp_files batch_test_ftp_files.py [-h] -ftp_folder_path FTP_FOLDER_PATH
                                                                                    -ftp_password FTP_PASSWORD
                                                                                    -predictions_cfg_json PREDICTIONS_CFG_JSON
                                                                                    [-download_folder_path DOWNLOAD_FOLDER_PATH]
                                                                                    [-csv_filename CSV_FILENAME]
                                                                                    [-max_runtime_minutes MAX_RUNTIME_MINUTES]
                                                                                    [-wait_time_minutes WAIT_TIME_MINUTES]
```


