import pandas as pd
import numpy as np
import pickle
import glob
import balanced_data_priority_2



def get_req_sounds(path_to_goertzel_components):


	pickle_files_1 = glob.glob(path_to_goertzel_components+'*.pkl')

	#removing the duplicate files if any
	pickle_files = list(set(pickle_files_1))
	print 'Number of  files :', len(pickle_files)

	#saving file names as a dataframe column
	ytid = []
	for each_file in pickle_files:
		ytid.append(eac_file.split('/')[-1][:11])
	arb_df = pd.DataFrame()
	arb_df['YTID'] = ytid


	#calling the balancing_dataset function to balance the data
	data= balancing_dataset.balanced_data()

	req_df=pd.DataFrame()
	req_df = data[['wav_file','YTID','labels_name','Data_dist_new']]

	#merge the datframes to get the finla dataframe with required columns
	df = pd.merge(req_df,arb_df,on='YTID')

	# seperate out the different sounds based on the column ''Data_dist_new'. Check for the seperating_different_sounds.py for more information about this column.
	#explosion sounds
	exp = df.loc[df['Data_dist_new'].apply(lambda arr: arr[0]==0)]
	print "explosion sound shape:", exp.shape
	#motor sounds
	mot = df.loc[df['Data_dist_new'].apply(lambda arr: arr[0]==1)]
	print "motor sounds shape :", mot.shape
	#nature sounds
	nat = df.loc[df['Data_dist_new'].apply(lambda arr: arr[0]==2)]
	print "nature sounds shape :", nat.shape
	#human sounds
	hum = df.loc[df['Data_dist_new'].apply(lambda arr: arr[0]==3)]
	print "human sounds shape :", hum.shape
	# wood sounds
	wod = df.loc[df['Data_dist_new'].apply(lambda arr: arr[0]==4)]
	print "wood sounds shape :", wod.shape
	# domestic animals sounds
	dom = df.loc[df['Data_dist_new'].apply(lambda arr: arr[0]==6)]
	print "domestic sounds shape :", dom.shape
	#tools sounds
	tools = df.loc[df['Data_dist_new'].apply(lambda arr: arr[0]==7)]
	print "tools sounds shape :", tools.shape
	#wild animals sounds
	wild = df.loc[df['Data_dist_new'].apply(lambda arr: arr[0]==5)]
	print "wild sounds shape :", wild.shape



	# return dataframe of each sounds seperately
	return mot,hum,wod,exp,dom,tools,wild,nat
