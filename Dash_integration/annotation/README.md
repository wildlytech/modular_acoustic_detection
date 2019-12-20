# Audio Annotation Tool

<br>

## Audio Annotation Tool
- Audio annotation tool enables you to do multiple things at a time for a single audio file as mentioned below using different tabs
	- **Spectrogram Tab** : Plots spectrogram of the ```.wav``` file
	- **Annotation Tab**: Layout to annotate audio ```.wav``` file and submit the annotations 
	- **Model Prediction Tab** : Predict the type of sounds present in ```.wav``` file using Machine Learning / Deep Learning Models
	- Learn more about Tabs and its usage from -   [Dash Tabs Documentation](https://dash.plot.ly/dash-core-components/tabs)
- Change the following lines of code to rename ```.csv``` file and labels that should be present in ```check box``` for easy annotation here [Line 28: annotation.py](https://github.com/wildlytech/modular_acoustic_detection/blob/9f95169ab4bf93e058b140bef07513c776de2190/Dash_integration/annotation/audio_annotation.py#L28)
```python
CSV_FILENAME = "New_annotation.csv"
CHECKLIST_DISPLAY = ["Nature", "Bird", "Wind", "Vehicle", "Honking", "Conversation"]
```
- To start running the local server and annotate files use the following command
```shell
$ python annotation.py
```
**Note**: Check for Link [http://127.0.0.1:8050/](http://127.0.0.1:8050/) from your browser to view & interact with the  application

###### Reference:
- You can check for screenshots of the application to see how it looks. 
	- Annotation Tab : [Annotation Tab Screenshot](https://drive.google.com/open?id=1MNRKR5pmgLUV7pd6EJ16XohuwpkOSi-B)
	- Model Prediction tab : [Prediction Tab Screenshot](https://drive.google.com/open?id=1cOj3LxgN-SyCHnZm_iwbdDBJGKVaOXBo)
	- Spectrogram Tab : [Spectrogram Tab Screenshot](https://drive.google.com/open?id=1TTsgsz6o08JhP0cgPYoZwPlXodqtRplo)



