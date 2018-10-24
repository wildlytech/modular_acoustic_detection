import glob
import argparse
import pickle
import pandas as pd

#parse the input path where the embeddings of the downloaded files are saved
PARSER = argparse.ArgumentParser(description='Input the path for embeddings of the downloaded files')
PARSER.add_argument('-path_to_embeddings', '--path_where_embedddings_saved', action='store', help='Input the path')
RESULT = PARSER.parse_args()

#Load the downloaded base dataframe generated from download_data() function of youtube_audioset
with open('downloaded_base_dataframe.pkl', 'rb') as f:
    UN_DF = pickle.load(f)

# glob all the generated pickle files( embeddings ) from the generting_embeddings.py
EMBEDDINGS_FILES = glob.glob(RESULT.path_where_embedddings_saved + '*.pkl')
#get the YTID from embedding files names
YTID = []
for each_file_name in EMBEDDINGS_FILES:
    YTID.append(each_file_name.split('/')[-1][:11])

#Reading all the pickle files into the python environment
EMBEDDINGS = []
print 'reading the embedding files..'
for embedding_each_file in EMBEDDINGS_FILES:
    with open(embedding_each_file, 'rb') as f:
        arb = pickle.load(f)
    EMBEDDINGS.append(arb)
    arb = []

#Creating a datafrane with features (embeddings) as column and then merge with the base DataFrame.
FEATURES_DF = pd.DataFrame()
FEATURES_DF['YTID'] = YTID
FEATURES_DF['features'] = EMBEDDINGS

#Merge the base dataframe with the intermediate dataframe.
#To get the final unbalanced dataframe with feature as column

FINAL_DF = pd.merge(UN_DF, FEATURES_DF, on='YTID')

#Now overwrite the unbalanced_data.pkl file with included features column
with open('downloaded_final_dataframe.pkl', 'w') as f:
    pickle.dump(FINAL_DF, f)

print 'Final data frame with feature column is saved as  " downloaded_final_dataframe.pkl "..!!'
