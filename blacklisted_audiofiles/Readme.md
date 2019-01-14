## Blacklisted Examples

We have blacklisted sounds that are having issue regarding with their labeling. 

We have categorized the blacklisted examples into certain categories. Two main categories out of them are as follows :

##### 1. Mislabeling : Sounds that are labelled wrongly. 
Eg. Some sounds are labelled as ```Silence``` While the audio isn't silence it has some sounds in it and also vice versa. 

##### 2.Under-labeling : All the sounds that are present in a audio clip are not labelled, only few among them are labelled.

Eg. Audio clip has ```[ Vehicle, Human ]``` in it but the labeling is done only as ```Human sounds```. 

##### Reading the above stored ```.wav``` file into python list. 

You  can read the above stored ```.txt``` into python as list values of ```.wav``` files using ```pickle``` library.

For Eg:
##### To open ```mislabelled_as_other_than_silence.txt``` this file into python list you have to run following commands:

```python
import pickle
with open("mislabelled_as_other_than_silence.txt", "rb") as file_object:
    mislabelled_list = pickle.load(file_object)
print "Mis-labelled Examples:", mislabelled_list
```
