import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import glob
import os
from youtube_audioset import get_csv_data

#Reading a pickle file consiting of the unbalanced data and sound labels
with open('unbalanced_data.pkl','rb') as f:
    df=pickle.load(f)
labels_csv=pd.read_csv('<path to class_labels_indices.csv>')
print('Files Loaded')
#Reading the coarse labels pickle file . It consists of seperate classes for sounds related to Explosion, Motor, Nature, Human, Wood with numbers assigned as 0,1,2,3,4 respectively.
with open('coarse_labels.pkl','rb') as f:
    lab=pickle.load(f)

#Taking the first 50000 examples
df=df.iloc[:][:]
df['positive_labels']=df['positive_labels'].apply(lambda arr: arr.split(','))
df['labels_name']=df['positive_labels'].map(lambda arr:[labels_csv.loc[labels_csv['mid']==x].display_name for x in arr])
df['labels_name']=df['labels_name'].apply(np.concatenate)
print(df['labels_name'])
df['Data_dist']=df['labels_name'].map(lambda arr:[lab.loc[lab['sounds']==y].id for y in arr])
df['start_seconds_int']=df['start_seconds'].apply(int)
df['YTID']=df['YTID'].astype(str) +'-'+df['start_seconds_int'].astype(str)

# Creating the Data distribution coulumn
df['Data_dist']=df['Data_dist'].apply(np.concatenate)
df['Data_dist']=df['Data_dist'].apply(set)
df['Data_dist']=df['Data_dist'].apply(list)
df['len_labels']=df['Data_dist'].apply(len)
print(df)
print(df.shape)
df['len_labels']=df['Data_dist'].apply(len)
print("Maximum number of labels for a sound:",df['len_labels'].max())

#For sounds with two labels in it. Taking each labels as a different sound
df1=df.loc[df['len_labels']==1]
d1=df1['Data_dist'].values.tolist()
print(len(d1))
e1=[]
n=0
for i in d1:
    e1.append(d1[n][0])
    n=n+1
print(len(e1))
df1['Data_dist']=e1
print(df1)
#For sounds with two labels in it
df2=df.loc[df['len_labels']==2]
d2=df2['Data_dist'].values.tolist()

print(len(d2))
e2_0=[]
e2_1=[]
for i in d2:
    e2_0.append(i[0])
    e2_1.append(i[1])
e2=e2_0+e2_1
print(len(e2))
df2=pd.concat([df2]*2, ignore_index=True)
df2['Data_dist']=e2
print(df2)
#For sounds with three labels in it
df3=df.loc[df['len_labels']==3]
d3=df3['Data_dist'].values.tolist()
print(len(d3))
e3_0=[]
e3_1=[]
e3_2=[]
for i in d3:
    e3_0.append(i[0])
    e3_1.append(i[1])
    e3_2.append(i[2])
e3=e3_0 + e3_1 + e3_2
print(len(e3))
df3=pd.concat([df3]*3, ignore_index=True)
df3['Data_dist']=e3
print(df3)
#For sounds with four lables in it
df4=df.loc[df['len_labels']==4]
d4=df4['Data_dist'].values.tolist()
print(len(d4))


req_df=pd.DataFrame()
req_df=pd.concat([df1,df2,df3],ignore_index=True)
print(req_df.shape)
print(req_df.head())
#Adding the Data_dist column to the DataFrame df and plotting the histogram of the data distribution
a=req_df['Data_dist'].value_counts(sort=True)
print(a)

#Saving all the classes of sounds ie Explosion, Motor, Nature, Human, Wood seperately.
#Explosion Sounds
df_exp=pd.DataFrame(req_df.loc[req_df['Data_dist']==0])
df_exp.index=range(a.iloc[2])
print(df_exp.shape)
print(df_exp.head())

with open('Explosion_sounds.pkl','w') as f:
    pickle.dump(df_exp,f)
#motor sounds
df_mot=pd.DataFrame(req_df.loc[req_df['Data_dist']==1])
df_mot.index=range(a.iloc[0])
print(df_mot.shape)
print(df_mot.head())
with open('Motor_sounds.pkl','w') as f:
    pickle.dump(df_mot,f)

#Nature sounds
df_nat=pd.DataFrame(req_df.loc[req_df['Data_dist']==2])
df_nat.index=range(a.iloc[1])
print(df_nat.shape)
print(df_nat.head())
with open('Nature_sounds.pkl','w') as f:
    pickle.dump(df_nat,f)

#Human sounds Balancing
df_hum=pd.DataFrame(req_df.loc[req_df['Data_dist']==3])
df_hum.index=range(a.iloc[3])
print(df_hum.shape)
print(df_hum.head())
with open('Human_sounds.pkl','w') as f:
    pickle.dump(df_hum,f)

#Wood sounds
df_wod=pd.DataFrame(req_df.loc[req_df['Data_dist']==4])
df_wod.index=range(a.iloc[4])
print(df_wod.shape)
print(df_wod.head())
with open('Wood_sounds.pkl','w') as f:
    pickle.dump(df_wod,f)
#plot the histogram to know the various amount of distrubution of sounds across the labels
req_df['Data_dist'].plot(kind='hist',xticks=[0,1,2,3,4])
plt.xlabel("Explosion,Motor,Nature,Human,Wood")
plt.show()
