# Audio Augmentation

<br>

#### 1. Audio Mixing - [Wikipedia Source](https://en.wikipedia.org/wiki/Audio_mixing)
- To mix two different audio clips both should be of ```equal Sampling Rates``` - [Sampling Rate?](https://en.wikipedia.org/wiki/Sampling_%28signal_processing%29)
- About Arguments:
	- ```-use_youtube_sounds``` : This argument should be passed followed by ```y``` if we need to mix Google audioset that are downloaded, otherwise please ignore this argument and the next two arguments as well i.e ```-type_one``` and ```-type_two```
	- ```-type_one``` : If first argument ```-use_youtube_sounds``` is passed with ```y``` then only pass this argument with any one type of label sounds to mix i.e whether ```motor```, ```motor```, ```explosion```, ```human```, ```nature```, ```domestic```, ```tools```
	- ```-type_two``` : If first argument ```-use_youtube_sounds``` is passed with ```y``` then only pass this argument with any one type of label sounds to mix i.e whether ```motor```, ```motor```, ```explosion```, ```human```, ```nature```, ```domestic```, ```tools```
	- ```-path_type_one_audio_files``` : If we need to mix audio files that are present in local disk not the Google audioset files, pass the path of that directory where all the wav files are present that are to mixed.
	- ```-path_type_two_audio_files``` : If we need to mix audio files that are present in local disk not the Google audioset files, pass the path of that directory where all the wav files are present that are to be mixed.
	- ```path_to_save_mixed_sounds``` : Path where to save mixed audio clips
- Follow the command below to mix audio files

```shell
$ python audio_mixing.py [-h]  -use_youtube_sounds
			       -type_one
			       -type_two
			       -path_type_two_audio_files
			       -path_type_two_audio_files
			       -path_to_save_mixed_sounds
```


