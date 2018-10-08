import pandas as pd
import numpy
import pickle
import glob

#give the path where the embeddings of the downloaded files are saved
path_where_embedddings_saved = 'data/audioset/explosion/'

#Load the downloaded base dataframe generated from download_data() function of youtube_audioset
with open('downloaded_base_dataframe.pkl','rb') as f:
    un_df=pickle.load(f)

# glob all the generated pickle files( embeddings ) from the generting_embeddings.py
embedding_files = glob.glob(path_where_embedddings_saved + '*.pkl')
#get the YTID from embedding files names
ytid =[]
for each_file_name in embedding_files:
    ytid.append(each_file_name.split('/')[-1][:11])

#Reading all the pickle files into the python environment
embeddings=[]
for embedding_each_file in embedding_files:
    with open(embedding_each_file,'rb') as f:
        arb=pickle.load(f)
    embeddings.append(arb)
    arb=[]

#Creating a datafrane with features ( embeddings ) as column and then merging with the base DataFrame.
features_df=pd.DataFrame()
features_df['YTID'] = ytid
features_df['features'] = embeddings

#Merge the base dataframe with the intermediate dataframe to get the final unbalanced dataframe with feature as column
final_df = pd.merge(un_df,features_df,on='YTID')

#Now overwrite the unbalanced_data.pkl file with included features column
with open('downloaded_final_dataframe.pkl','w') as f:
    pickle.dump(final_df,f)

print 'Final data frame with feature column is saved as  " downloaded_final_dataframe.pkl "..!!'
