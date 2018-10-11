# Prediction on Audio-files 

### Prediction on Audiomoth recorded files 

Audiomoth device is used to record the audio clips from various locations/field. The trained multi-label model is run on these sounds to validate the detection accuracy on real world sounds. 
<br>
we use pre-trained weights of ```multiclass_model.py``` which will be in ```.h5``` file for predictions.
<br>

##### 1. To run predictions on audiomoth recordings. ( .WAV files )

```
$ python predict_on_audiomoth_files.py [-h] [ -path_to_audio_files PATH_TO_AUDIO_FILES ] [ -path_to_embeddings PATH_TO_EMBEDDINGS ]
```
###### output returns :
- ```audiomoth_prediction.csv``` file is created with one of the column as ```predicted_labels```
