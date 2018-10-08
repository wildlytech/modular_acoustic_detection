#Importing necessary library funtions
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import glob
import os


#Reading a pickle file consiting of the unbalanced data and sound labels
with open('downloaded_final_dataframe.pkl','rb') as f:
    un=pickle.load(f)
labels_csv=pd.read_csv('data/audioset/class_labels_indices.csv')
print('Files Loaded')
print un.shape


#Reading the coarse labels pickle file . It consists of seperate classes for sounds related to Explosion, Motor, Nature, Human, Wood with numbers assigned as 0,1,2,3,4 respectively.
lab=pd.read_csv('coarse_labels.csv')

# Creating the Data distribution coulumn
un['Data_dist_new']=un['labels_name'].map(lambda arr:[lab.loc[lab['sounds']==y].id for y in arr])
un['Data_dist_new']=un['Data_dist_new'].apply(np.concatenate)
un['Data_dist_new']=un['Data_dist_new'].apply(set)
un['Data_dist_new']=un['Data_dist_new'].apply(list)


# get the dataframe for having single type of sounds
un['len'] = un['Data_dist_new'].apply(len)


#For with single sound , two , three , four and five
un_1 = un.loc[un['len']==1]
un_2 = un.loc[un['len']==2]
un_3 = un.loc[un['len']==3]
un_4 = un.loc[un['len']==4]
un_5 = un.loc[un['len']==5]


# seperate out the sounds from single labelled. Check out the coarse_labels.csv file to know what are index values for Explosion, Motor, Natutre, Human, and all other sounds
pure_exp = un_1.loc[un_1['Data_dist_new'].apply(lambda arr: arr[0]==0)]
pure_mot = un_1.loc[un_1['Data_dist_new'].apply(lambda arr: arr[0]==1)]
pure_nat = un_1.loc[un_1['Data_dist_new'].apply(lambda arr: arr[0]==2)]
pure_hum = un_1.loc[un_1['Data_dist_new'].apply(lambda arr: arr[0]==3)]
pure_wod = un_1.loc[un_1['Data_dist_new'].apply(lambda arr: arr[0]==4)]
pure_wild = un_1.loc[un_1['Data_dist_new'].apply(lambda arr: arr[0]==5)]
pure_dom = un_1.loc[un_1['Data_dist_new'].apply(lambda arr: arr[0]==6)]
pure_tools = un_1.loc[un_1['Data_dist_new'].apply(lambda arr: arr[0]==7)]

# write out all those pure single labelled Sounds
file_names = ['pure_exp','pure_mot','pure_nat','pure_hum','pure_wod','pure_wild','pure_dom','pure_tools']
for i,j in zip(file_names,[pure_exp,pure_mot,pure_nat,pure_hum,pure_wod,pure_wild,pure_dom,pure_tools]):
    with open('diff_class_datasets/new_wild/'+i+'_'+str(j.shape[0])+'_wild.pkl','w') as f:
        pickle.dump(j,f)



# seperate out sounds which are Multi-labelled with two classes
#exp with other sounds
exp_mot = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==0) & (arr[1]==1)))]
exp_nat = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==0) & (arr[1]==2)))]
exp_hum = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==0) & (arr[1]==3)))]
exp_wod = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==0) & (arr[1]==4)))]
exp_wild = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==0) & (arr[1]==5)))]
exp_dom = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==0) & (arr[1]==6)))]
exp_tools = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==0) & (arr[1]==7)))]
# exp_mot = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: (arr[0]==0 & arr[1]==8)]

# write out all those Sounds
file_names = ['exp_mot','exp_nat','exp_hum','exp_wod','exp_wild','exp_dom','exp_tools']
for i,j in zip(file_names,[exp_mot,exp_nat,exp_hum,exp_wod,exp_wild,exp_dom,exp_tools]):
    with open('diff_class_datasets/new_wild/'+i+'_'+str(j.shape[0])+'_wild.pkl','w') as f:
        pickle.dump(j,f)


# mot with other sounds
mot_nat = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==1) & (arr[1]==2)))]
mot_hum = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==1) & (arr[1]==3)))]
mot_wod = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==1) & (arr[1]==4)))]
mot_wild = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==1) & (arr[1]==5)))]
mot_dom = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==1) & (arr[1]==6)))]
mot_tools = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==1) & (arr[1]==7)))]

# write out all those Sounds
file_names = ['mot_nat','mot_hum','mot_wod','mot_wild','mot_dom','mot_tools']
for i,j in zip(file_names,[mot_nat,mot_hum,mot_wod,mot_wild,mot_dom,mot_tools]):
    with open('diff_class_datasets/new_wild/'+i+'_'+str(j.shape[0])+'_wild.pkl','w') as f:
        pickle.dump(j,f)


#nature with other Sounds
nat_hum = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==2) & (arr[1]==3)))]
nat_wod = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==2) & (arr[1]==4)))]
nat_wild = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==2) & (arr[1]==5)))]
nat_dom = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==2) & (arr[1]==6)))]
nat_tools = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==2) & (arr[1]==7)))]
file_names = ['nat_hum','nat_wod','nat_wild','nat_dom','nat_tools']
for i,j in zip(file_names,[nat_hum,nat_wod,nat_wild,nat_dom,nat_tools]):
    with open('diff_class_datasets/new_wild/'+i+'_'+str(j.shape[0])+'_wild.pkl','w') as f:
        pickle.dump(j,f)


#human sounds with other Sounds
hum_wod = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==3) & (arr[1]==4)))]
hum_wild = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==3) & (arr[1]==5)))]
hum_dom = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==3) & (arr[1]==6)))]
hum_tools = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==3) & (arr[1]==7)))]
file_names = ['hum_wod','hum_wild','hum_dom','hum_tools']
for i,j in zip(file_names,[hum_wod,hum_wild,hum_dom,hum_tools]):
    with open('diff_class_datasets/new_wild/'+i+'_'+str(j.shape[0])+'_wild.pkl','w') as f:
        pickle.dump(j,f)


#wood and other sounds
wod_wild = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==4) & (arr[1]==5)))]
wod_dom = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==4) & (arr[1]==6)))]
wod_tools = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==4) & (arr[1]==7)))]

# write out all those Sounds
file_names = ['wod_wild','wod_dom','wod_tools']
for i,j in zip(file_names,[wod_wild,wod_dom,wod_tools]):
    with open('diff_class_datasets/new_wild/'+i+'_'+str(j.shape[0])+'_wild.pkl','w') as f:
        pickle.dump(j,f)


#wild and other sounds
wild_dom = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==5) & (arr[1]==6)))]
wild_tools = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==5) & (arr[1]==7)))]
file_names = ['wild_dom','wild_tools']

# write out all those Sounds
for i,j in zip(file_names,[wild_dom,wild_tools]):
    with open('diff_class_datasets/new_wild/'+i+'_'+str(j.shape[0])+'_wild.pkl','w') as f:
        pickle.dump(j,f)


#domestic and other
dom_tools = un_2.loc[un_2['Data_dist_new'].apply(lambda arr: ((arr[0]==6) & (arr[1]==7)))]
with open('diff_class_datasets/new_wild/'+'dom_tools_'+str(dom_tools.shape[0])+'_wild.pkl','w') as f:
    pickle.dump(dom_tools,f)

#Sounds with more than 2 classes labelled are witten
print 'three labelled sounds shape: ', un_3.shape
with open('3_labelled_priority1_'+str(un_3.shape[0])+'.pkl' ,'w') as f:
    pickle.dump(un_3,f)
print 'four labelled sounds shape: ', un_4.shape
with open('4_labelled_priority1_'+str(un_4.shape[0])+'.pkl' ,'w') as f:
    pickle.dump(un_4,f)
print 'five labelled sounds shape: ', un_5.shape
with open('5_labelled_prioirty1_'+str(un_5.shape[0])+'.pkl' ,'w') as f:
    pickle.dump(un_5,f)
