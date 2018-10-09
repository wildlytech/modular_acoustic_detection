from youtube_audioset import download_data

if __name__ == "__main__":	
	#specify the target sounds to download and set the path for downlaoded files to store 
	target_sounds = youtube_audioset.ambient_sounds + youtube_audioset.impact_sounds 
	target_path = 'sounds/ambient_impact/'

	#start downloading all the files 
	#It will also return a base dataframe with [ YTID, start_seconds, end_seconds, positive_labels, labels_name, wav_file ] name  as columns along with audio wav files 
	download_data(target_sounds, target_path)
