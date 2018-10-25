"""
Seperates different sounds based on its labelling
"""
#Importing necessary library funtions
import pickle
import os
import argparse
import numpy as np
import pandas as pd



#parse the input path to write the dataframes of seprated sounds
PARSER = argparse.ArgumentParser(description='Input path to write the dataframe \
                                 of seperated sounds Make it interpretable directory name \
                                 [ eg "diff_class_datasets/explosion_sounds" for explosion sound].\
                                 so that it could be easily accessed ')
PARSER.add_argument('-path_to_write_different_sounds',
                    '--path_to_write_different_sounds',
                    action='store',
                    help='Input the path')
RESULT = PARSER.parse_args()

#create a path if not there
if not os.path.exists(RESULT.path_to_write_different_sounds):
    os.makedirs(RESULT.path_to_write_different_sounds)


#Reading a pickle file consiting of the unbalanced data and sound labels
with open('downloaded_final_dataframe.pkl', 'rb') as f:
    UN = pickle.load(f)
LABELS_CSV = pd.read_csv('data/audioset/class_labels_indices.csv')
print 'Files Loaded'
print UN.shape


#Reading the coarse labels pickle file.
#It consists of seperate classes for sounds related to
#Explosion, Motor, Nature, Human, Wood with numbers assigned as 0,1,2,3,4 respectively.
LAB = pd.read_csv('coarse_labels.csv')

# Creating the Data distribution coulumn
UN['Data_dist_new'] = UN['labels_name'].map(lambda arr: [LAB.loc[LAB['sounds'] == y].id for y in arr])
UN['Data_dist_new'] = UN['Data_dist_new'].apply(np.concatenate)
UN['Data_dist_new'] = UN['Data_dist_new'].apply(set)
UN['Data_dist_new'] = UN['Data_dist_new'].apply(list)


# get the dataframe for having single type of sounds
UN['len'] = UN['Data_dist_new'].apply(len)


#For with single sound , two , three , four and five
UN_1 = UN.loc[UN['len'] == 1]
UN_2 = UN.loc[UN['len'] == 2]
UN_3 = UN.loc[UN['len'] == 3]
UN_4 = UN.loc[UN['len'] == 4]
UN_5 = UN.loc[UN['len'] == 5]


# seperate out the sounds from single labelled.
#Check out the coarse_labels.csv file
#to know what are index values for Explosion, Motor, Natutre, Human, and all other sounds
PURE_EXP = UN_1.loc[UN_1['Data_dist_new'].apply(lambda arr: arr[0] == 0)]
PURE_MOT = UN_1.loc[UN_1['Data_dist_new'].apply(lambda arr: arr[0] == 1)]
PURE_NAT = UN_1.loc[UN_1['Data_dist_new'].apply(lambda arr: arr[0] == 2)]
PURE_HUM = UN_1.loc[UN_1['Data_dist_new'].apply(lambda arr: arr[0] == 3)]
PURE_WOD = UN_1.loc[UN_1['Data_dist_new'].apply(lambda arr: arr[0] == 4)]
PURE_WILD = UN_1.loc[UN_1['Data_dist_new'].apply(lambda arr: arr[0] == 5)]
PURE_DOM = UN_1.loc[UN_1['Data_dist_new'].apply(lambda arr: arr[0] == 6)]
PURE_TOOLS = UN_1.loc[UN_1['Data_dist_new'].apply(lambda arr: arr[0] == 7)]

# write out all those pure single labelled Sounds
FILE_NAMES = ['pure_exp', 'pure_mot', 'pure_nat', 'pure_hum',
              'pure_wod', 'pure_wild', 'pure_dom', 'pure_tools']
for i, j in zip(FILE_NAMES,
                [PURE_EXP, PURE_MOT, PURE_NAT, PURE_HUM,
                 PURE_WOD, PURE_WILD, PURE_DOM, PURE_TOOLS]):
    with open(RESULT.path_to_write_different_sounds+i+'_'+str(j.shape[0])+'.pkl', 'w') as f:
        pickle.dump(j, f)



# seperate out sounds which are Multi-labelled with two classes
#exp with other sounds
EXP_MOT = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 0) & (arr[1] == 1)))]
EXP_NAT = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 0) & (arr[1] == 2)))]
EXP_HUM = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 0) & (arr[1] == 3)))]
EXP_WOD = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 0) & (arr[1] == 4)))]
EXP_WILD = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 0) & (arr[1] == 5)))]
EXP_DOM = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 0) & (arr[1] == 6)))]
EXP_TOOLS = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 0) & (arr[1] == 7)))]
# exp_MOT = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: (arr[0]==0 & arr[1]==8)]

# write out all those Sounds
FILE_NAMES = ['exp_mot', 'exp_nat', 'exp_hum', 'exp_wod', 'exp_wild', 'exp_dom', 'exp_tools']
for i, j in zip(FILE_NAMES, [EXP_MOT, EXP_NAT, EXP_HUM, EXP_WOD, EXP_WILD, EXP_DOM, EXP_TOOLS]):
    with open(RESULT.path_to_write_different_sounds+i+'_'+str(j.shape[0])+'.pkl', 'w') as f:
        pickle.dump(j, f)


# mot with other sounds
MOT_NAT = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 1) & (arr[1] == 2)))]
MOT_HUM = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 1) & (arr[1] == 3)))]
MOT_WOD = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 1) & (arr[1] == 4)))]
MOT_WILD = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 1) & (arr[1] == 5)))]
MOT_DOM = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 1) & (arr[1] == 6)))]
MOT_TOOLS = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 1) & (arr[1] == 7)))]

# write out all those Sounds
FILE_NAMES = ['mot_nat', 'mot_hum', 'mot_wod', 'mot_wild', 'mot_dom', 'mot_tools']
for i, j in zip(FILE_NAMES, [MOT_NAT, MOT_HUM, MOT_WOD, MOT_WILD, MOT_DOM, MOT_TOOLS]):
    with open(RESULT.path_to_write_different_sounds+i+'_'+str(j.shape[0])+'.pkl', 'w') as f:
        pickle.dump(j, f)


#nature with other Sounds
NAT_HUM = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 2) & (arr[1] == 3)))]
NAT_WOD = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 2) & (arr[1] == 4)))]
NAT_WILD = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 2) & (arr[1] == 5)))]
NAT_DOM = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 2) & (arr[1] == 6)))]
NAT_TOOLS = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 2) & (arr[1] == 7)))]
FILE_NAMES = ['nat_hum', 'nat_wod', 'nat_wild', 'nat_dom', 'nat_tools']
for i, j in zip(FILE_NAMES, [NAT_HUM, NAT_WOD, NAT_WILD, NAT_DOM, NAT_TOOLS]):
    with open(RESULT.path_to_write_different_sounds+i+'_'+str(j.shape[0])+'.pkl', 'w') as f:
        pickle.dump(j, f)


#human sounds with other Sounds
HUM_WOD = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 3) & (arr[1] == 4)))]
HUM_WILD = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 3) & (arr[1] == 5)))]
HUM_DOM = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 3) & (arr[1] == 6)))]
HUM_TOOLS = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 3) & (arr[1] == 7)))]
FILE_NAMES = ['hum_wod', 'hum_wild', 'hum_dom', 'hum_tools']
for i, j in zip(FILE_NAMES, [HUM_WOD, HUM_WILD, HUM_DOM, HUM_TOOLS]):
    with open(RESULT.path_to_write_different_sounds+i+'_'+str(j.shape[0])+'.pkl', 'w') as f:
        pickle.dump(j, f)


#wood and other sounds
WOD_WILD = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 4) & (arr[1] == 5)))]
WOD_DOM = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 4) & (arr[1] == 6)))]
WOD_TOOLS = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 4) & (arr[1] == 7)))]

# write out all those Sounds
FILE_NAMES = ['wod_wild', 'wod_dom', 'wod_tools']
for i, j in zip(FILE_NAMES, [WOD_WILD, WOD_DOM, WOD_TOOLS]):
    with open(RESULT.path_to_write_different_sounds+i+'_'+str(j.shape[0])+'.pkl', 'w') as f:
        pickle.dump(j, f)


#wild and other sounds
WILD_DOM = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 5) & (arr[1] == 6)))]
WILD_TOOLS = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 5) & (arr[1] == 7)))]
FILE_NAMES = ['wild_dom', 'wild_tools']

# write out all those Sounds
for i, j in zip(FILE_NAMES, [WILD_DOM, WILD_TOOLS]):
    with open(RESULT.path_to_write_different_sounds+i+'_'+str(j.shape[0])+'.pkl', 'w') as f:
        pickle.dump(j, f)


#domestic and other
DOM_TOOLS = UN_2.loc[UN_2['Data_dist_new'].apply(lambda arr: ((arr[0] == 6) & (arr[1] == 7)))]
with open(RESULT.path_to_write_different_sounds+'dom_tools_'+
          str(DOM_TOOLS.shape[0])+'.pkl', 'w') as f:
    pickle.dump(DOM_TOOLS, f)

#Sounds with more than 2 classes labelled are witten
print 'three labelled sounds shape: ', UN_3.shape
with open(RESULT.path_to_write_different_sounds+'3_labelled_priority1_'+
          str(UN_3.shape[0])+'.pkl', 'w') as f:
    pickle.dump(UN_3, f)
print 'four labelled sounds shape: ', UN_4.shape
with open(RESULT.path_to_write_different_sounds+'4_labelled_priority1_'+
          str(UN_4.shape[0])+'.pkl', 'w') as f:
    pickle.dump(UN_4, f)
print 'five labelled sounds shape: ', UN_5.shape
with open(RESULT.path_to_write_different_sounds+'5_labelled_prioirty1_'+
          str(UN_5.shape[0])+'.pkl', 'w') as f:
    pickle.dump(UN_5, f)
