## Goertzel Filter Implementation

We implemented the goertzel algorithm in order to filter out only particular target frequencies of each sound files. This is done in order to do a basic detection on board using the goertzel frequency components of sound files ( .wav files )

##### Goertzel Algorithm example : 

##### 1. To run the test example for implementing the goertzel algorithm 

```
$ python goertzel_algorithm.py 
```

###### output of above script returns :
Goertzel frequency component of ```speech.wav```. In this case ```TARGET_FREQUENCY``` is set to ```400Hz```. you can change to see the different components variation.

##### 2. Generate the goertzel frequency components of each audio file 
We have selected the four frequency components that holds good for detecting the ```Motor sounds, Human sounds, Explosion sounds, Domestic sounds, Tools``` in an audio file. Those four frequency components are ```[800Hz, 1600Hz, 2000Hz, 2300Hz]``` for downsampled audio to sampling rate of ```8000 samples/second```

```
$ python goertzel_filter_components [-h] -audio_files_path AUDIO_FILES_PATH -path_to_freq_comp PATH_TO_FREQ_COMP
```

###### output of above script returns :
- generates the frequeny components of each audio file in specified directory and store as ```.pkl``` files in given directory.

##### 3. Detection of Impact Vs Ambient using goertzel frequency components
```
$python goertzel_detection_model.py [-h] -path_to_goertzel_components PATH_TO_GOERTZEL_COMPONENTS
```

###### output returns:
- ```.h5``` file having the trained weights of the model.

