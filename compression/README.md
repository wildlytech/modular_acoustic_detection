# Audio compression 

#### Audio encoders 

Compression for audio files may be Lossy or Lossless. Here, we tried out different types of audio compressions using different encoders.
<br>
<br>
##### 1. **To compress the audio files using different audio encoders** :
This script will take the audio files and try compress the audio using specified audio encoders. This example code take ```libopus``` audio encoder for compressing ```wav``` file. This script also requires ```balancing_dataset.py``` to be in same directory.

```
$ python compression.py [-h] [-audio_files_path AUDIO_FILES_PATH] [-path_to_write_compressed_files PATH_TO_WRITE_COMPRESSED_FILES] 
```
###### output returns :
- compressed audio files at specified directory . It may be ``` .opus, .mp2, .flac, ,ac3, .m4a, .wma, .wv, .aac, .ogg ``` and many such.  

##### Reference :
- The results for different compression techniques comparing ```speed, size of compressed file, bitrate etc```  are in  [Audio-encoders-spreadsheet](https://docs.google.com/spreadsheets/d/1rJYaTsp8JnH0bdWxCY30T40-iMi_kbX43oiQ87PGy90/edit?usp=sharing).
- The results for performance of different sounds on different compression techniques are in [codecs-on-sounds](https://docs.google.com/spreadsheets/d/1g9GbqOsAYhmdfHXLeXvc4S6hp-N1KOPgZWNxwKV1y_s/edit?usp=sharing).
<br>
<br>

##### 2. To decompress the compressed audio files 
Decompressing the audio files back to ```wav``` file will result in some loss if the compression used is Lossy if not ideally there will be no loss. 
<br> 
We are decompressing back, audio files to ```wav``` files to check the performance of the detection models ( ```binary_model.py``` and ```multiclass_model.py``` ) on the decompressed ```wav``` files.
```
$ python decmpression.py [-h] [-compressed_files_path COMPRESSED_FILES_PATH] [-path_to_write_decomp_files PATH_TO_WRITE_DECOMP_FILES]
```
###### output returns :
-Decompressed ```wav``` files in specified directory. 
<br>
<br>

##### 3. To test the latency of different compression techniques :
This gives the time taken for compressing the ```wav``` file into different compressed file formats.
```
$ python different_compression_latency.py 
```
###### output returns :
- Prints out the time taken by each audio enocder to compress test wav file ```speech.wav```




