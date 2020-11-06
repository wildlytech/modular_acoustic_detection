# Getting Audio Data
<br>

## From Different Sources

##### 1. From  [Google Audioset](https://research.google.com/audioset/) 
-  Download only Sounds of Interest
	- This will download the sounds ```( .wav files )``` that you are interested in, from enlisted sounds of . you can see the list for class of sounds by using ```[-h]``` argument for script in command line .

``` $ python download_soi.py [-h] -target_sounds -target_path```

- Downloading whole sound Library
	- This is a really lengthy process and hence is not advisable unless you absolutely have to.
	- This will download all the ```Impact``` and ```Ambient ``` sounds from Google Audioset from the definition that we defined in the [youtube_audioset.py](https://github.com/wildlytech/modular_acoustic_detection/blob/bce293f40520baf4967646f67a19918a144b0f3e/youtube_audioset.py#L160)

```$ python download_all_sounds.py ```

<br>

##### 2. From AudioMoth device : [AudioMoth Recording Device](https://www.openacousticdevices.info/)
- About AudioMoth : [AudioMoth Recording Device](https://www.openacousticdevices.info/)
- We use this recording device to collect audio files ```(.wav)``` of required audio length / duration.
- If  AudioMoth files are already annotated in ```.csv``` file then we could use [data_preprocessing.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/get_data/data_preprocessing.py)  script to get the dataframe ```TYPE 2``` from it and use these files are training / testing ML models
- To get the ```TYPE 2 ``` base dataframe from an annotated ```.csv``` file, follow the command below
- About Argument:
	-  ```-annotation_file``` : Path for an annotated ```.csv``` file
	-  ```-path_for_saved_embeddings``` : Path (Directory path) where all the embeddings for the wav files are saved ```(.pkl)```. If embeddings are not present it can be ignored
	-  ```-path_to_save_dataframe``` : Path along with name with ```.pkl``` extension to write ```TYPE 2``` dataframe

```shell
$ python data_preprocessing.py  -annotation_file
			        -path_for_saved_embeddings
			        -path_to_save_dataframe
```

<br>

##### 3. From YouTube (Scraping Audio Files):

- We can scrape the required sounds from online platforms like YouTube and other sources. But here we are scraping sounds from YouTube platform
- Change the below constants in [youtube_scrape.py](https://github.com/wildlytech/modular_acoustic_detection/blob/master/get_data/youtube_scrape.py) script as per the requirement
```python 
SEARCH_KEYWORD = "dog barking"
PATH_TO_WRITE_AUDIO = "youtube_scraped_audio/"
NUMBER_AUDIOCLIPS_LIMIT = 10
``` 
- Once the changes are made as per the requirement follow the command below to run the script
```shell
$ python youtube_script.py
```
#### 4. From Xenocanto website

- We can scrape required sounds from the xenocanto website and generate a dataframe for training
- In order to do this we run the xenocanto_to_dataframe.py file in the shell
- About arguments:
    - Required arguments:
        - ```-bird_species``` : Input bird species by separating name with space and enclosed within quotes
        - ```-output_path```: Path to save generated data
  - Optional arguments:
    - ```-delete```: Delete downloaded audio files after script execution
    - ```-dont_delete```: Don't delete downloaded audio files after script execution

```shell
$ python xenocanto_to_dataframe.py  -bird_species
			        -output_path
			        -delete
                    -dont_delete
```
