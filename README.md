# Modular Acoustic Detection
<br>

## 1. Environment Setup

#### Recommended System Requirements:
##### Local Hosting Setup:
- 2 Core CPU / 4 GiB RAM
- OS: Ubuntu 18:04
##### Cloud Hosting on AWS
- AWS Instance: t*.medium or above
- AWS AMI: ami-0b84c6433cdbe5c3e (Ubuntu 18:04)

#### 1.1 Local Repository setup

- #####  Clone the Repository
**IMPORTANT**: Make sure to add GitHub SSH Keys to your local system before cloning the repository.\
Guide: [Add SSH key to GitHub](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

```shell
$ git clone git@github.com:wildlytech/modular_acoustic_detection.git
```

- #####  To load all git sub modules :
```shell
# Change to directory
$ cd modular_acoustic_detection

$ git submodule update --init --recursive
```
- ##### To download all the data files :
```shell
# Make script executable
$ chmod 777 download_data_files.sh

$ ./download_data_files.sh
```

#### 1.2 Python environment Setup

##### Approach 1:
**Note**: Recommended to install Anaconda to manage the different environment and to avoid package/library version conflicts.\
Download: [Anaconda](https://www.anaconda.com/distribution/)

- ##### Create a separate environment with Python 3.7 Version
```shell
$ conda create -n env_name python=3.7
```
**Note**: Change ```env_name``` to your convenient name

- ##### Activate the created environment
```shell
$ conda activate env_name
```
**Note**: After successful activation of the environment terminal should display something similar to above
```shell
(env_name)$
```

##### Approach 2:
**Note**: This method is an alternate to using Anaconda and to instead use Python virtualenv environments.

- ##### Install required Ubuntu Packages
```shell
# Make script executable
$ chmod 777 ubuntu_packages_install.sh

# Run script to install
$ ./ubuntu_packages_install.sh
```

- ##### Create virtual environment with Python 3.7 Version
```shell
$ virtualenv -p python3.7 env_name
```
**Note**: Change ```env_name``` to your convenient name

- ##### To load your virtual environment
```shell
$ cd env_name

# activate the virtual environment
$ source bin/activate
```

#### **Important:** Install Python requirements for either approach:
- To install all the required library python packages at one go. Type in the command mentioned below

```shell
$ pip install -r requirements.txt
```

<br>

## 2. Getting Audio Data
- This process is to get the audio files ```(.wav format)``` from YouTube which are labelled by Google
- We can more details about the data-set and its annotation on the mentioned link. [Google Audioset](https://research.google.com/audioset/)
- For getting wav files from the above mentioned source we have enter to the ```get_data/``` directory.

Follow the command to navigate into ```get_data/```
```shell
$ cd get_data/
```
**Note**: [Navigate to get_data/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/get_data)

<br>

## 3.  Audio Augmentation
- Audio Augmentation is a process wherein we  perform operations like mixing multiple sounds, Time shift the audio data, Scale the audio, change the volume of the audio etc to make it audible differently  than the original sound but also at the same making sure it is realistic
- To perform operations related augmentation navigate to [augmentation/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/augmentation)
- Follow the command in terminal to navigate to augmentation directory
- ``` $ cd augmentation/ ```

<br>

## 4. Generate Embeddings
This will download the embeddings as ```.pkl``` files at the directory where you specify. This script requires additional functional scripts found at [Tensorflow-models-repo](https://github.com/tensorflow/models/tree/9b57f41ce21cd7264c52140c9ab31cdfc5169fcd/research/audioset).

```shell
$ python generating_embeddings.py   --wav_file
                    --path_to_write_embeddings
```
###### Output of above script will return :
- Embeddings in ```.pkl``` files for each downloaded audio file at specified directory. (```--wav_file``` requires the directory path where ```.wav``` files are saved )

<br>

## 5. Create Base Dataframe
- This will add the generated embedding values of each audio file to base dataframe columns if it already exists ```(TYPE 1)```, otherwise it creates base dataframe with ```["wav_file", "features"]``` columns i.e ```(TYPE 2)```. Final dataframe will now have one extra column when compared with ```downloaded_base_dataframe.pkl``` i.e with ```["features"]```
- About Arguments:
    -  ```-dataframe_without_feature``` : If ```TYPE 1``` dataframe exists already, path of it should be given otherwise it can be ignored
    -  ```-path_for_saved_embeddings``` : Directory path where all the ```.pkl``` files are saved
    -  ```-path_to_write_dataframe``` : Path along with name of the file with ```.pkl``` extension to write the final dataframe
```shell
$ python create_base_dataframe.py [-h] -dataframe_without_feature
                       -path_for_saved_embeddings
                       -path_to_write_dataframe
```
###### Output of this script will return :
-  If  ```-dataframe_without_feature``` ```(TYPE 1)``` dataframe is inputted then ```["features]``` column is added to same dataframe, if not a new dataframe with ```["wav_file", "features"]``` columns is stored

<br>

## 6.  Separating Sounds Based on Labels
- This script will read the dataframe ```(TYPE 1)``` i.e it should have ```["labels_name"]``` column in it ,  separates out the sounds based on labeling, creates a different dataframe as per labels and writes at target path given.
- You can check [coarse_labels.csv](https://github.com/wildlytech/modular_acoustic_detection/blob/master/coarse_labels.csv) file to know the mapping of the labels and the separation of each sounds takes place
- To separate sounds based on labels navigate to [data_preprocessing_cleaning/separating_different_sounds.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/data_preprocessing_cleaning/seperating_different_sounds.py)
- Follow the command below to navigate to the directory and execute the script to separate sounds
- ``` $ cd data_preprocessing_cleaning/ ```

<br>

## 7.  Training ML/DL Models
- Once we have required audioset consisting of different labelled audio clips ```labels_name``` each of 10 seconds and their appropriate embeddings ```features```  in a dataframe format (preferably) we can use these dataframe files for training ML / DL models
- Any Dataframe file with these columns in it can be included into training data i.e ```["wav_file", "labels_name", "features]```
- We have to add the path of that required dataframe (Includes above mentioned columns) in [balanced_data.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/balancing_dataset.py)
- Once the path of the required datframe is placed in above mentioned script navigate to ```models/``` to train different types of ML and DL models
- To navigate follow the command below
```shell
$ cd models/
```
**Note**: [Navigate to models/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/models)

<br>

## 8. Predicting on Audio files using trained ML/DL models
- This folder consists of scripts used to predict audio files using different types of ML/DL trained models
- After training ML/DL models we will able to save the trained model weights in ```.h5``` file for each model. To predict any audio file we will be using these model weights file to make predictions
- Navigate to ```predictions/``` folder to start predictng on single/multiple audio files using different models
- To navigate follow the command below
```shell
$ cd predictions/
```
**Note**: [Navigate to predictions/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/predictions)

<br>

## 9. Compressing & Decompressing of Audio Files
- Definition of Audio Compression: [Wikipedia Source](https://en.wikipedia.org/wiki/Audio_compression_(data))
- Two different types of Audio compression
     -  Lossy Audio Compression
     - Lossless Audio Compression

To perform various types of audio compression techniques & decompressing back the compressed audio files navigate to ```compression/``` directory. To navigate follow the command below
```shell
$ cd compression/
```
**Note**: [Navigate to compression/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/compression)

<br>

## 10. Goertzel Algorithm

-   Definition of Goertzel Filter: [Wikipedia Source](https://en.wikipedia.org/wiki/Goertzel_algorithm)
- Navigate to goertzel_filter/ directory if you want to :
    - Visualize the audio file in spectrogram after applying goertzel filter
    - Extract  particular frequency components of an audio file
- To navigate to ```goertzel_filter/``` directory follow the below command
```shell
$ cd goertzel_filter/
```
**Note**: [Navigate to goertzel_filter/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/goertzel_filter)

<br>

## 11. Dash User Interface Applications
- About Dash Framework: [Dash | Plotly](https://plot.ly/dash/)
- We have used Dash framework for building local web apps for different purposes stated below
    - **Audio Annotation** : We can annotate audio files (.wav format) in any folder present locally and save all the annotations in ```.csv file```. It also enables to view spectrogram and see the model's prediction for that wavfile
        - To annotate audio files navigate to [Dash_integration/annotation/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/Dash_integration/annotation)
        - Follow the command to navigate to that folder in terminal
        - ``` $ cd Dash_integraion/annotation/```
    - **Device Report** : Enables to see generate a concise report for each device that is uploading files in FTP server. Device parameters such as Battery performance, Location Details etc can be visualized using this app
        - To generate report navigate to  [Dash_integration/device_report/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/Dash_integration/device_report)
        - Follow the command to navigate to this folder in terminal
        - ``` $ cd Dash_integraion/device_report/```
    - **Monitoring and Alert** : Enables user to monitor FTP server directories, Device(s), get alert based on detection of any sounds of interest, upload multiple audio wavfiles to see the predictions etc
        - To monitor and get alerts via SMS navigate to [Dash_integration/monitoring_alert/](https://github.com/wildlytech/modular_acoustic_detection/tree/master/Dash_integration/monitoring_alert)
        - Follow the command in terminal to navigate to this
        - ``` $ cd Dash_integration/monitoring_alert/ ```