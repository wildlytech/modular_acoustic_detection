"""
Returns Balanced dataframe mainly to Train data
"""
import pickle
import random
import pandas as pd

# Pure Sounds datapath
YOUTUBE_PURE_DATAPATH = 'diff_class_datasets/Datasets/pure/'
AUDIOMOTH_PURE_DATAPATH = "audiomoth_different_split/pure/"

# Mixed sounds data path
AUDIOMOTH_MIXED_DATAPATH = "audiomoth_different_split/mixed_sounds/"
YOUTUBE_MIXED_DATAPATH_AUGMENTED = "/media/wildly/Seagate/Embeddings_Aud_Mix/"


def distinguished_audiomoth_sounds():
    """
    Files with single class : AudioMoth Source
    """

    ###########################################################################
    # Pure motor sounds - AudioMoth (Different Locations)
    ###########################################################################
    with open(AUDIOMOTH_PURE_DATAPATH + "pure_mot/pure_mot_1017_Nandi_hills_2_datafile_labels_emb.pkl", 'rb') as file_obj:
        ad_mot1 = pickle.load(file_obj)
    ad_mot1 = ad_mot1.sample(1000)

    with open(AUDIOMOTH_PURE_DATAPATH+"pure_mot/pure_mot_3173_Nandi_hills_1_datafile_labels_emb.pkl", 'rb') as file_obj:
        ad_mot2 = pickle.load(file_obj)
    ad_mot2 = ad_mot2.sample(1500)

    with open(AUDIOMOTH_PURE_DATAPATH +"pure_mot/pure_mot_224_gbs_am1_chainsaw_data_labels.pkl", 'rb') as file_obj:
        ad_mot3 = pickle.load(file_obj)
    ad_mot3 = ad_mot3.sample(224)

    with open(AUDIOMOTH_PURE_DATAPATH+"pure_mot/pure_mot_210_gbs_am2_chainsaw_data_labels.pkl", 'rb') as file_obj:
        ad_mot4 = pickle.load(file_obj)
    ad_mot4 = ad_mot4.sample(210)

    ###########################################################################
    # Pure Human sounds : AudioMoth (Different Locations)
    ###########################################################################

    with open(AUDIOMOTH_PURE_DATAPATH+"pure_hum/pure_hum_149_gbs_am2_chainsaw_data_labels.pkl", 'rb') as file_obj:
        ad_hum1 = pickle.load(file_obj)
    with open(AUDIOMOTH_PURE_DATAPATH+"pure_hum/pure_hum_142_gbs_am1_chainsaw_data_labels.pkl", 'rb') as file_obj:
        ad_hum2 = pickle.load(file_obj)

    ###########################################################################
    # Pure Nature Sounds - AudioMoth (Different Locations)
    ###########################################################################
    with open(AUDIOMOTH_PURE_DATAPATH+"pure_nat/pure_nat_3216_GBS_waynad_datafile_labels_emb.pkl", 'rb') as file_obj:
        ad_nat1 = pickle.load(file_obj)
    ad_nat1 = ad_nat1.sample(3000)
    with open(AUDIOMOTH_PURE_DATAPATH+"pure_nat/pure_nat_2077_Erivikulam_datafile_labels_emb.pkl", 'rb') as file_obj:
        ad_nat2 = pickle.load(file_obj)
    ad_nat2 = ad_nat2.sample(2000)
    with open(AUDIOMOTH_PURE_DATAPATH+"pure_nat/pure_nat_922_gbs_am2_chainsaw_data_labels.pkl", 'rb') as file_obj:
        ad_nat3 = pickle.load(file_obj)
    ad_nat3 = ad_nat3.sample(700)

    ###########################################################################
    # Pure Tools Sounds : AudioMoth (Different Locations)
    ###########################################################################
    with open(AUDIOMOTH_PURE_DATAPATH+ "pure_tools/pure_tools_520_gbs_am2_chainsaw_data_labels.pkl", 'rb') as file_obj:
        ad_tool = pickle.load(file_obj)
    ad_tool = ad_tool.sample(500)
    with open(AUDIOMOTH_PURE_DATAPATH+"pure_tools/pure_tools_672_gbs_am1_chainsaw_data_labels.pkl", 'rb') as file_obj:
        ad_tool1 = pickle.load(file_obj)
    ad_tool1 = ad_tool1.sample(500)

    ###########################################################################
    # Concatenate all the respective pure sounds
    ###########################################################################

    ad_motor_pure = pd.concat([ad_mot1, ad_mot2], ignore_index=True)
    ad_nature_pure = pd.concat([ad_nat1, ad_nat2, ad_nat3], ignore_index=True)
    ad_tool_pure = pd.concat([ad_tool, ad_tool1], ignore_index=True)
    ad_human_pure = pd.concat([ad_hum1, ad_hum2], ignore_index=True)

    return ad_motor_pure, ad_human_pure, ad_nature_pure, ad_tool_pure


def include_mixed_sounds(mixed_sounds_flag):
    """
    Including Mixed data also in the dataset
    """
    if mixed_sounds_flag:

        #######################################################################
        # Check for Motor and Human - 7500 : Youtube and AudioMoth
        #######################################################################

        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"hum_mix_mot_9532.pkl", 'rb') as file_obj:
            df_hm = pickle.load(file_obj)
        df_hm = df_hm.sample(2000)

        with open(AUDIOMOTH_MIXED_DATAPATH+"mot_hum/mot_hum_422_Nandi_hills_1_datafile_labels_emb.pkl", 'rb') as file_obj:
            ad_hm1 = pickle.load(file_obj)
        ad_hm1 = ad_hm1.sample(421)

        with open(AUDIOMOTH_MIXED_DATAPATH+"mot_hum/mot_hum_135_Nandi_hills_2_datafile_labels_emb.pkl", 'rb') as file_obj:
            ad_hm2 = pickle.load(file_obj)
        ad_hm2 = ad_hm2.sample(134)

        df_mixed_motor_human = pd.concat([df_hm, ad_hm1, ad_hm2], ignore_index=True)

        #######################################################################
        # Check for Nature Mixed sounds : Youtube and AudioMoth
        #######################################################################
        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"hum_mix_nat_with_wavfiles_9532.pkl", 'rb') as file_obj:
            df_hn = pickle.load(file_obj)
        df_hn = df_hn.sample(2500)

        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"motor_mixed_nature.pkl", 'rb') as file_obj:
            df_mn = pickle.load(file_obj)
        df_mn = df_mn.sample(2500)

        with open(AUDIOMOTH_MIXED_DATAPATH+"mot_nat/mot_nat_1017_Nandi_hills_1_datafile_labels_emb.pkl", 'rb') as file_obj:
            ad_mn1 = pickle.load(file_obj)[:]
        ad_mn1 = ad_mn1.sample(1000)

        with open(AUDIOMOTH_MIXED_DATAPATH+"mot_nat/mot_nat_149_Nandi_hills_2_datafile_labels_emb.pkl", 'rb') as file_obj:
            ad_mn2 = pickle.load(file_obj)[:]
        ad_mn2 = ad_mn2.sample(100)

        with open(AUDIOMOTH_MIXED_DATAPATH+"mot_nat/mot_nat_54_gbs_am1_chainsaw_data_labels.pkl", 'rb') as file_obj:
            ad_mn3 = pickle.load(file_obj)[:]
        ad_mn3 = ad_mn3.sample(50)

        df_mixed_nature = pd.concat([df_hn, df_mn, ad_mn1, ad_mn2, ad_mn3], ignore_index=True)

        #######################################################################
        # Check for sounds Mixed with Explosion : Youtube
        #######################################################################
        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"exp_mix_mot_7957.pkl", 'rb') as file_obj:
            df_me = pickle.load(file_obj)
        df_me = df_me.sample(2500)

        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"nature_mixed_explosion.pkl", 'rb') as file_obj:
            df_ne = pickle.load(file_obj)
        df_ne = df_ne.sample(2500)

        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"human_mixed_explosion.pkl", 'rb') as file_obj:
            df_he = pickle.load(file_obj)
        df_he = df_he.sample(2500)

        df_mixed_explosion = pd.concat([df_me, df_ne, df_he], ignore_index=True)

        #######################################################################
        # Check for Mixed with Tools : Youtube and AudioMoth
        #######################################################################
        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"dom_mixed_tools_7789_wavfiles.pkl", 'rb') as file_obj:
            df_dt = pickle.load(file_obj)
        df_dt = df_dt.sample(2500)

        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"human_mixed_tools.pkl", 'rb') as file_obj:
            df_ht = pickle.load(file_obj)
        df_ht = df_ht.sample(2500)

        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"motor_mixed_tools.pkl", 'rb') as file_obj:
            df_mt = pickle.load(file_obj)
        df_mt = df_mt.sample(2500)

        df_mixed_tools = pd.concat([df_dt, df_ht, df_mt], ignore_index=True)

        #######################################################################
        # Check for Mixed with Domestic : Youtube
        #######################################################################
        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"dom_mixed_tools_7789_wavfiles.pkl", 'rb') as file_obj:
            df_td = pickle.load(file_obj)
        df_td = df_td.sample(2500)

        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"human_mixed_domestic.pkl", 'rb') as file_obj:
            df_hd = pickle.load(file_obj)
        df_hd = df_hd.sample(2500)

        with open(YOUTUBE_MIXED_DATAPATH_AUGMENTED+"motor_mixed_domestic.pkl", 'rb') as file_obj:
            df_md = pickle.load(file_obj)
        df_md = df_md.sample(2500)

        df_mixed_domestic = pd.concat([df_td, df_hd, df_md], ignore_index=True)

        #######################################################################
        # concatenate sounds
        #######################################################################

        df_mixed = pd.concat([df_mixed_motor_human,
                              df_mixed_explosion,
                              df_mixed_nature,
                              df_mixed_tools,
                              df_mixed_domestic], ignore_index=True)

        return df_mixed


def balanced_data(audiomoth_flag, mixed_sounds_flag):
    """
    Function to read all data frames and balancing
    """

    ###########################################################################
    # Files with single class : Youtube
    ###########################################################################
    with open(YOUTUBE_PURE_DATAPATH+'Explosion/pure_exp_7957.pkl', 'rb') as file_obj:
        pure_exp = pickle.load(file_obj)
    with open(YOUTUBE_PURE_DATAPATH+'Motor/pure_mot_76045.pkl', 'rb') as file_obj:
        pure_mot = pickle.load(file_obj)
    with open(YOUTUBE_PURE_DATAPATH+'Human_sounds/pure_hum_46525.pkl', 'rb') as file_obj:
        pure_hum = pickle.load(file_obj)
    with open(YOUTUBE_PURE_DATAPATH+'Wood/pure_wod_1115.pkl', 'rb') as file_obj:
        pure_wod = pickle.load(file_obj)
    with open(YOUTUBE_PURE_DATAPATH+'Nature_sounds/pure_nat_13527.pkl', 'rb') as file_obj:
        pure_nat = pickle.load(file_obj)
    with open(YOUTUBE_PURE_DATAPATH+'Domestic/pure_dom_9497.pkl', 'rb') as file_obj:
        pure_dom = pickle.load(file_obj)
    with open(YOUTUBE_PURE_DATAPATH+'Tools/pure_tools_8113.pkl', 'rb') as file_obj:
        pure_tools = pickle.load(file_obj)
    # with open(YOUTUBE_PURE_DATAPATH+'pure/Wild/pure_wild_7061.pkl','rb') as file_obj:
    #     pure_wild=pickle.load(file_obj)

    ###########################################################################
    # Balancing and experimenting
    ###########################################################################
    exp = pd.concat([pure_exp], ignore_index=True)
    mot = pd.concat([pure_mot], ignore_index=True)
    hum = pd.concat([pure_hum], ignore_index=True)
    nat = pd.concat([pure_nat], ignore_index=True)
    dom = pd.concat([pure_dom], ignore_index=True)
    tools = pd.concat([pure_tools], ignore_index=True)
    # wood= pd.concat([pure_wod],ignore_index=True)
    # wild = pd.concat([pure_wild[:300]],ignore_index=True)

    ###########################################################################
    # Shuffling the data
    ###########################################################################
    mot_req = mot.loc[random.sample(list(range(0, mot.shape[0])), 2000)]
    exp_req = exp.loc[random.sample(list(range(0, exp.shape[0])), 2000)]
    hum_req = hum.loc[random.sample(list(range(0, hum.shape[0])), 2000)]
    nat_req = nat.loc[random.sample(list(range(0, nat.shape[0])), 2000)]
    dom_req = dom.loc[random.sample(list(range(0, dom.shape[0])), 2000)]
    tools_req = tools.loc[random.sample(list(range(0, tools.shape[0])), 2000)]
    # wood_req = wood.loc[random.sample(range(0,wood.shape0]), wood.shape[0])]
    # wild_req = wild.loc[random.sample(range(0,wild.shape[0]),1000)]

    df_pure_sounds_youtube = pd.concat([mot_req, exp_req, hum_req, nat_req, dom_req, tools_req], ignore_index=True)

    ###########################################################################
    # Check to include audiomoth data or not
    ###########################################################################
    if audiomoth_flag == 1:
        audiomoth_nature, audiomoth_motor, audiomoth_hum, audiomoth_tool = distinguished_audiomoth_sounds()
        sounds_to_concatenate = [df_pure_sounds_youtube+audiomoth_nature+audiomoth_motor+audiomoth_hum+audiomoth_tool]
    else:
        sounds_to_concatenate = [df_pure_sounds_youtube]

    ###########################################################################
    # Check to include mixed sounds or not
    ###########################################################################
    if mixed_sounds_flag == 1:
        df_mixed_sounds = include_mixed_sounds(1)
        sounds_to_concatenate = sounds_to_concatenate + [df_mixed_sounds]
    else:
        pass

    ###########################################################################
    # concat the required sounds
    ###########################################################################
    data_frame = pd.concat(sounds_to_concatenate, ignore_index=True)
    print('Final dataframe shape :', data_frame.shape)

    ###########################################################################
    # All labels should be lowercase for comparison and identification
    ###########################################################################
    data_frame['labels_name'] = data_frame.labels_name.apply(lambda arr: [x.lower() for x in arr])

    ###########################################################################
    # Free up the space
    ###########################################################################
    del pure_nat, pure_dom, pure_exp, pure_mot, pure_wod, pure_tools
    return data_frame


###############################################################################
# Main Function
###############################################################################
if __name__ == "__main__":
    AUDIOMOTH_FLAG = 1
    MIXED_SOUNDS_FLAG = 1
    DATA_FRAME = balanced_data(AUDIOMOTH_FLAG, MIXED_SOUNDS_FLAG)
