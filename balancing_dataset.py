import pandas as pd
import pickle
import numpy as np
import os
import random

def balanced_data():

    # Reding all the files into pandas dataframe

    #Files with single class
    with open('diff_class_datasets/Datasets/pure/Explosion/pure_exp_7957.pkl','rb') as f:
        pure_exp=pickle.load(f)
    with open('diff_class_datasets/Datasets/pure/Motor/pure_mot_76045.pkl','rb') as f:
        pure_mot=pickle.load(f)
    with open('diff_class_datasets/Datasets/pure/Human_sounds/pure_hum_46525.pkl','rb') as f:
        pure_hum=pickle.load(f)
    with open('diff_class_datasets/Datasets/pure/Wood/pure_wod_1115.pkl','rb') as f:
        pure_wod=pickle.load(f)
    with open('diff_class_datasets/Datasets/pure/Nature_sounds/pure_nat_13527.pkl','rb') as f:
        pure_nat=pickle.load(f)
    with open('diff_class_datasets/Datasets/pure/Domestic/pure_dom_9497.pkl','rb') as f:
        pure_dom=pickle.load(f)
    with open('diff_class_datasets/Datasets/pure/Tools/pure_tools_8113.pkl','rb') as f:
        pure_tools=pickle.load(f)
    # with open('diff_class_datasets/Datasets/pure/Wild/pure_wild_7061.pkl','rb') as f:
    #     pure_wild=pickle.load(f)

    # Mixed sounds
    with open('diff_class_datasets/Datasets/Mixed_2/mixed_with_human/hum_exp_3969.pkl','rb') as f:
        hum_exp=pickle.load(f)
    with open('diff_class_datasets/Datasets/Mixed_2/mixed_with_wod/wood_tools_1283.pkl','rb') as f:
        wood_tool=pickle.load(f)
    with open('diff_class_datasets/Datasets/Mixed_2/mixed_with_nature/nat_dom_345.pkl','rb') as f:
        nat_dom=pickle.load(f)
    with open('diff_class_datasets/Datasets/Mixed_2/mixed_with_motor/mot_wod_62.pkl','rb') as f:
        mot_wod=pickle.load(f)

     #Balancing and experimenting
    exp = pd.concat([pure_exp],ignore_index=True)
    mot = pd.concat([pure_mot],ignore_index=True)
    hum = pd.concat([pure_hum],ignore_index=True)
    # wood= pd.concat([pure_wod],ignore_index=True)
    nat = pd.concat([pure_nat],ignore_index=True)
    dom = pd.concat([pure_dom],ignore_index=True)
    tools = pd.concat([pure_tools],ignore_index=True)
    # wild = pd.concat([pure_wild[:300]],ignore_index=True)

    # Shuffling the data
    mot_req = mot.loc[random.sample(range(0,mot.shape[0]),7900)]
    exp_req = exp.loc[random.sample(range(0,exp.shape[0]),7900)]
    hum_req = hum.loc[random.sample(range(0,hum.shape[0]),7900)]
    # wood_req = wood.loc[random.sample(range(0,wood.shape0]),wood.shape[0])]
    nat_req = nat.loc[random.sample(range(0,nat.shape[0]),7900)]
    dom_req = dom.loc[random.sample(range(0,dom.shape[0]),7900)]
    tools_req = tools.loc[random.sample(range(0,tools.shape[0]),7900)]
    # wild_req = wild.loc[random.sample(range(0,wild.shape[0]),1000)]


    #concat the required sounds
    df = pd.concat([exp_req,nat_req,hum_req,mot_req,dom_req,tools_req],ignore_index=True)
    print 'Final dataframe shape :',df.shape

    #Free up the space
    del pure_nat,pure_dom,pure_exp, pure_mot, pure_wod, pure_tools, exp_req, nat_req, hum_req, mot_req, dom_req, tools_req

    return df

if __name__=="__main__":
    df = balanced_data()
