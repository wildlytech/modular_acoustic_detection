"""
Returns the dataframe of different sound labels
"""
import glob
import sys
sys.path.insert(0, '../')
import pandas as pd
import balancing_dataset


#################################################################################
        # Helper Function
#################################################################################
def get_req_sounds(path_to_goertzel_components):
    """
    Returns the dataframes with required sounds
    """
    pickle_files = glob.glob(path_to_goertzel_components+'*.pkl')

    # removing the duplicate files if any
    pickle_files = list(set(pickle_files))
    print 'Number of  files :', len(pickle_files)

    # saving file names as a dataframe column
    ytid = []
    for each_file in pickle_files:
        ytid.append(each_file.split('/')[-1][:11])
    arb_data_frame = pd.DataFrame()
    arb_data_frame['YTID'] = ytid

    # calling the balancing_dataset function to balance the data
    data = balancing_dataset.balanced_data(audiomoth_flag=0, mixed_sounds_flag=0)
    data = data[['wav_file', 'YTID', 'labels_name', 'Data_dist_new']]

    # merge the datframes to get the finla dataframe with required columns
    data_frame = pd.merge(data, arb_data_frame, on='YTID')


    #################################################################################
       # seperate different sounds based on 'Data_dist_new' column
    #################################################################################


    #################################################################################
                       # seperate Explosion sounds
    #################################################################################
    exp = data_frame.loc[data_frame['Data_dist_new'].apply(lambda arr: arr[0] == 0)]
    print "explosion sound shape:", exp.shape


    #################################################################################
                        # seperate motor sounds
    #################################################################################
    mot = data_frame.loc[data_frame['Data_dist_new'].apply(lambda arr: arr[0] == 1)]
    print "motor sounds shape :", mot.shape


    #################################################################################
                        # seperate nature sounds
    #################################################################################
    nat = data_frame.loc[data_frame['Data_dist_new'].apply(lambda arr: arr[0] == 2)]
    print "nature sounds shape :", nat.shape


    #################################################################################
                        # seperate human sounds
    #################################################################################
    hum = data_frame.loc[data_frame['Data_dist_new'].apply(lambda arr: arr[0] == 3)]
    print "human sounds shape :", hum.shape


    #################################################################################
                        # seperate wood sounds
    #################################################################################
    wod = data_frame.loc[data_frame['Data_dist_new'].apply(lambda arr: arr[0] == 4)]
    print "wood sounds shape :", wod.shape


    #################################################################################
                        # seperate domestic animals sounds
    #################################################################################
    dom = data_frame.loc[data_frame['Data_dist_new'].apply(lambda arr: arr[0] == 6)]
    print "domestic sounds shape :", dom.shape


    #################################################################################
                        # seperate tools sounds
    #################################################################################
    tools = data_frame.loc[data_frame['Data_dist_new'].apply(lambda arr: arr[0] == 7)]
    print "tools sounds shape :", tools.shape



    #################################################################################
                        # seperate wild animals sounds
    #################################################################################
    wild = data_frame.loc[data_frame['Data_dist_new'].apply(lambda arr: arr[0] == 5)]
    print "wild sounds shape :", wild.shape

    # return dataframe of each sounds seperately
    return mot, hum, wod, exp, dom, tools, wild, nat
