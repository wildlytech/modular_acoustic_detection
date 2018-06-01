import pandas as pd
import numpy
import pickle
import glob


#Load the unbalanced_data.pkl file
with open('unbalanced_data.pkl','rb') as f:
    un_df=pickle.load(f)

#Glob all the generated pickle files( embeddings ) from the generting_embeddings.py
pickle_files = glob.glob('<path where the embeddings are written>/*.pkl')


#Reading all the pickle files into the python environment
embeddings=[]
for pkl in pickle_files:
    with open(pkl,'rb') as f:
        arb=pickle.load(f)
    embeddings.append(arb)
    arb=[]

#Creating a datafrane with features ( embeddings ) as column and then merging with the base DataFrame.
features_df=pd.DataFrame()
features_df['pkl_files']=pickle_files
fetaures_df['features']=embeddings

#Merge the base dataframe with the intermediate dataframe to get the final unbalanced dataframe with feature as column
un_df['pkl_files'] = un_df['YTID'].astype(str)+'-'+un_df['start_seconds'].apply(int).astype(str)+'.pkl'
final_df=pd.merge(un_df,fetaures_df,on='pkl_files')
del final_df['pkl_files']

#Now overwrite the unbalanced_data.pkl file with included features column
with open('unbalanced_data.pkl','w') as f:
    pickle.dump(final_df,f)
