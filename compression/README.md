# Audio compression

<br>

#### 1. Audio Compression and Audio Encoders
-   Definition of Audio Compression: [Wikipedia Source](https://en.wikipedia.org/wiki/Audio_compression_(data))
	-   Two different types of Audio compression
		-  Lossy Audio Compression
		-   Lossless Audio Compression
- Definition of Audio Codecs : [Wikipedia Source](https://en.wikipedia.org/wiki/Audio_codec)
	- Different types of Audio encoders that are supported by scripts below are : [aac, ac3, mp2, flac, libopus](https://github.com/wildlytech/modular_acoustic_detection/blob/b9e76653ae2cf59550767e4acabb63bfca8e0748/compression/compression.py#L21)

<br>

***
#### 2.  Compressing the audio files using different audio encoders :
- This script will take the audio files and try compress the audio using specified audio encoders.
- Different types of Audio encoders that are supported by scripts below are : [aac, ac3, mp2, flac, libopus](https://github.com/wildlytech/modular_acoustic_detection/blob/b9e76653ae2cf59550767e4acabb63bfca8e0748/compression/compression.py#L21)
- About Arguments:
	- ```-path_to_original_audio_files``` : Directory path to the location where all the ```.wav``` files are present, which are to be compressed
	- ```-path_to_compressed_audio_files``` : Directory path to the location where all the compressed audio files should be written
	- ```-codec_type``` : Specify any one type of audio encoder to compress in required format. Available codecs type [aac, ac3, mp2, flac, libopus](https://github.com/wildlytech/modular_acoustic_detection/blob/b9e76653ae2cf59550767e4acabb63bfca8e0748/compression/compression.py#L21)
- Follow the command below to start compressing audio files using required audio codec

```
$ python -m compression.compression [-h]   -path_to_original_audio_files
                                           -codec_type
                                           -path_to_write_compressed_files
```

###### Reference :
- The results for different compression techniques comparing ```speed, size of compressed file, bitrate etc```  are in  [Audio-encoders-spreadsheet](https://docs.google.com/spreadsheets/d/1rJYaTsp8JnH0bdWxCY30T40-iMi_kbX43oiQ87PGy90/edit?usp=sharing).
- The results for performance of different sounds on different compression techniques are in [codecs-on-sounds](https://docs.google.com/spreadsheets/d/1g9GbqOsAYhmdfHXLeXvc4S6hp-N1KOPgZWNxwKV1y_s/edit?usp=sharing).

<br>

***
####  2. Decompressing the compressed audio files
- Decompressing the audio files back to ```.wav``` format will result in some loss if the compression used is Lossy.
- We are decompressing back, audio files to ```wav``` files to check the performance of the detection models ( ```Binary Relevance Models``` and ```Multilabel Models``` ) on the decompressed ```.wav``` files.
- About arguments :
	- ```-path_to_compressed_audio_files``` : Directory Path where all the compressed audio files are present
	- ```-path_to_decompressed_audio_files``` : Directory path to the location where decompressed ```.wav``` files are to be stored
	- ```-codec_type``` : Type of compressed files present at ```-path_to_compressed_audio_files``` path
```
$ python -m compression.decmpression [-h]  -path_to_compressed_audio_files
                                           -path_to_decompressed_audio_files
                                           -codec_type
```
###### output returns :
- Decompressed ```.wav``` files in specified directory.

<br>

***
#### 4. Testing  latency of different compression techniques :
- This gives the time taken for compressing the ```speech.wav``` file into different compressed file formats.
- ```speech.wav``` is test audio file that is uploaded to the repository : [Speech.wav](https://github.com/wildlytech/modular_acoustic_detection/blob/master/compression/speech.wav)
```
$ python -m compression.different_compressions_latency
```
###### output  :
- Prints out the time taken by each audio enocder to compress test wav file ```speech.wav```





