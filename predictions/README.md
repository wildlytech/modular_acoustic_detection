## Predictions on sounds 

### Prediction on Audiomoth recorded files 

Audiomoth device used to record the audio clips from various locations/field. The trained multi-label model is run on these sounds to validate the detection accuracy on real world sounds. 
<br>
we use pre-trained weights of ```multiclass_model.py``` which will be in ```.h5``` file for predictions.
<br>

##### 1. To run predictions on audiomoth recordings. ( .WAV files )

```
$ python predict_on_audiomoth_files.py [-h] [ -path_to_audio_files PATH_TO_AUDIO_FILES ] [ -path_to_embeddings PATH_TO_EMBEDDINGS ]
```
###### output returns :
- ```audiomoth_prediction.csv``` file is created with one of the column as ```predicted_labels```

##### 2.To run predictions on single .wav file :
you can do it by two ways: 
<br>
- ##### shell script 
```shell
# Make script executable
$ chmod 777 predict_wav_file.sh 

# Run script with the wav file along with path as argument
$ ./predict_wav_file   speech.wav
```
- ##### python executable script 
```shell
$ python predict_on_wav_file.py  --wav_file speech.wav
```
###### output returns:
- prints out the output predictions of the input wavfile in our case ```speech.wav```.

