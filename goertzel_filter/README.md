# Goertzel Algorithm

- Definition of Goertzel Filter: [Wikipedia Source](https://en.wikipedia.org/wiki/Goertzel_algorithm)
- We implemented the goertzel algorithm in order to filter out only particular target frequencies of each sound files.
- This is done in order to do a basic ```Binary``` detection on board using the goertzel frequency components of sound files ( ```.wav``` files)

<br>

***
#### 1. Test example  of the Goertzel algorithm
- About Arguments:
    - ```-target_frequency_to_filter``` : Select the integer value of the target frequency to filter in ```Hz``` it should be below ```4000Hz```  (according to Nyquist frequency) as we are downsampling audio file to ```8000Hz```
    - ```-wavfile_path_to_filter``` : Path of an audio file (```.wav``` file) to filter
- Follow the command below to execute Goertzel algorithm on single audio file
```
$ python -m goertzel_filter.goertzel_algorithm [-h] -target_frequency_to_filter
                                                    -wavfile_path_to_filter
```
###### output :
- Goertzel frequency component of given wav file are generated .
- Plots ```spectrogram``` for the original wav file and filtered frequency components v/s Time

<br>

***
#### 2. Generate Filter components for Batch of Files
- We have selected the four frequency components that holds good for detecting the ```Motor sounds, Human sounds, Explosion sounds, Domestic sounds, Tools``` in an audio file.
- Those four frequency components are ```[800Hz, 1600Hz, 2000Hz, 2300Hz]``` for downsampled audio to sampling rate of ```8000Hz```
- About Arguments :
    - ```-audio_files_path``` : Directory path of ```.wav``` files for which Goertzel filtering needs to implemented
    - ```-path_to_freq_comp``` : Directory path to write the filter components of each audio file in ```.pkl``` extension files
- Parameters that we can change / play around in snippet are below. They are all in ```Hz``` units
```python
TARGET_FREQUENCIES = [800, 1600, 2000, 2300]
ACCEPTABLE_SAMPLINGRATE = 48000
DOWNSAMPLING_FREQUENCY = 8000
```
- Follow the command below to start generating filter components for audio files
```
$ python -m goertzel_filter.goertzel_filter_components [-h] -audio_files_path
                                                            -path_to_freq_comp
```
###### Output:
- Generates the frequency components of each audio file in specified directory and store as ```.pkl``` files.

<br>

***
#### 3. Training Binary Goertzel Model using frequency components
 - This is similar to training a simple binary model
 - Prerequisite : Balancing the data is the prerequisite which can be made in [balancing_dataset_goertzel.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/goertzel_filter/balancing_dataset_goertzel.py) script
- Toggle the flag in below snippet of  [goertzel_detection_model.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/goertzel_filter/goertzel_detection_model.py) if we need include audiomoth sounds, mixed sounds as well. By default only Google audioset is used.
```python
DATA_FRAME = balancing_dataset_goertzel.balanced_data(audiomoth_flag=0, mixed_sounds_flag=0)
```
- To train a simple binary model follow the command shown below
```shell
$ python -m goertzel_filter.goertzel_detection_model
```
###### output :
- ```.h5``` file having the trained weights of the model.
- ```.pkl``` file having all the weights of each layer separately in a list


