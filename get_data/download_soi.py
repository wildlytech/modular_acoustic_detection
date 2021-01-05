"""
Downloads Sounds of Interest audio files amoung the list
"""
import argparse
from subprocess import check_call, CalledProcessError
import threading
import pickle
import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
from sklearn.preprocessing import LabelBinarizer



###########################################################################
                    #Define the Sounds
###########################################################################

EXPLOSION_SOUNDS = [
    'Fireworks',
    'Burst, pop',
    'Eruption',
    'Gunshot, gunfire',
    'Explosion',
    'Boom',
    'Fire'
]

MOTOR_SOUNDS = [
    'Chainsaw',
    'Medium engine (mid frequency)',
    'Light engine (high frequency)',
    'Heavy engine (low frequency)',
    'Engine starting',
    'Engine',
    'Motor vehicle (road)',
    'Vehicle'
]

WOOD_SOUNDS = [
    'Wood',
    'Chop',
    'Splinter',
    'Crack'
]

HUMAN_SOUNDS = [
    'Chatter',
    'Conversation',
    'Speech',
    'Narration, monologue',
    'Babbling',
    'Whispering',
    'Laughter',
    'Chatter',
    'Crowd',
    'Hubbub, speech noise, speech babble',
    'Children playing',
    'Whack, thwack',
    'Smash, crash',
    'Breaking',
    'Crushing',
    'Tearing',
    'Run',
    'Walk, footsteps',
    'Clapping'

]


DOMESTIC_SOUNDS = [
    'Dog',
    'Bark',
    'Howl',
    'Bow-wow',
    'Growling',
    'Bay',
    'Livestock, farm animals, working animals',
    'Yip',
    'Cattle, bovinae',
    'Moo',
    'Cowbell',
    'Goat',
    'Bleat',
    'Sheep',
    'Squawk',
    'Domestic animals, pets'

]


TOOLS_SOUNDS = [
    'Jackhammer',
    'Sawing',
    'Tools',
    'Hammer',
    'Filing (rasp)',
    'Sanding',
    'Power tool'
]


WILD_ANIMALS = [
    'Roaring cats (lions, tigers)',
    'Roar',
    'Bird',
    'Bird vocalization, bird call, bird song',
    'Squawk',
    'Pigeon, dove',
    'Chirp, tweet',
    'Coo',
    'Crow',
    'Caw',
    'Owl',
    'Hoot',
    'Gull, seagull',
    'Bird flight, flapping wings',
    'Canidae, dogs, wolves',
    'Rodents, rats, mice',
    'Mouse',
    'Chipmunk',
    'Patter',
    'Insect',
    'Cricket',
    'Mosquito',
    'Fly, housefly',
    'Buzz',
    'Bee, wasp, etc.',
    'Frog',
    'Croak',
    'Snake',
    'Rattle'
]

NATURE_SOUNDS = [
    "Silence",
    "Stream",
    "Wind noise (microphone)",
    "Wind",
    "Rustling leaves",
    "Howl",
    "Raindrop",
    "Rain on surface",
    "Rain",
    "Thunderstorm",
    "Thunder",
    'Crow',
    'Caw',
    'Bird',
    'Bird vocalization, bird call, bird song',
    'Chirp, tweet',
    'Owl',
    'Hoot'

]



###########################################################################
            #Description and Help
###########################################################################

DESCRIPTION = 'Input one of these sounds : explosion_sounds , wood_sounds , motor_sounds,\
               human_sounds, tools ,domestic_sounds, Wild_animals, nature_sounds'
HELP = 'Input the target sounds. It should be one of the listed sounds'




###########################################################################
    #Defining Ambient and Impact sounds as to what sounds it must comprise of.
###########################################################################

AMBIENT_SOUNDS = NATURE_SOUNDS
IMPACT_SOUNDS = EXPLOSION_SOUNDS + WOOD_SOUNDS + MOTOR_SOUNDS + \
                HUMAN_SOUNDS + TOOLS_SOUNDS + DOMESTIC_SOUNDS



###########################################################################
        #create a dictionary of sounds
###########################################################################

SOUNDS_DICT = {'explosion_sounds': EXPLOSION_SOUNDS, 'wood_sounds': WOOD_SOUNDS,
               'nature_sounds': NATURE_SOUNDS, 'motor_sounds': MOTOR_SOUNDS,
               'human_sounds': HUMAN_SOUNDS, 'tools': TOOLS_SOUNDS,
               'domestic_sounds': DOMESTIC_SOUNDS, 'Wild_animals':WILD_ANIMALS}




###########################################################################
        #parse the input arguments given from command line
###########################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-target_sounds', '--target_sounds', action='store',
                    help=HELP, default='explosion_sounds')
PARSER.add_argument('-target_path', '--target_path', action='store',
                    help='Input the path', default='sounds/explosion_sounds/')
RESULT = PARSER.parse_args()


def get_csv_data(target_sounds):
    """
    Read all the csv file data into pandas dataframe
    """
    label_names = pd.read_csv("data/audioset/class_labels_indices.csv")

    balanced_train = pd.read_csv("data/audioset/balanced_train_segments.csv",
                                 quotechar='"',
                                 skipinitialspace=True,
                                 skiprows=2)
    balanced_train.columns = [balanced_train.columns[0][2:]] + \
                              balanced_train.columns[1:].values.tolist()

    unbalanced_train = pd.read_csv("data/audioset/unbalanced_train_segments.csv",
                                   quotechar='"',
                                   skipinitialspace=True,
                                   skiprows=2)
    unbalanced_train.columns = [unbalanced_train.columns[0][2:]] + \
                                unbalanced_train.columns[1:].values.tolist()

    train = pd.concat([unbalanced_train, balanced_train], axis=0, ignore_index=True)

    # Filter out sounds we don't care about.
    forest_sounds = pd.Series(target_sounds, name='display_name')
    forest_sounds = pd.DataFrame(forest_sounds).merge(label_names,
                                                      how='inner',
                                                      left_on='display_name',
                                                      right_on='display_name')

    # Binarizer for labels of interest.  We add ;None' as a placeholder for sounds that
    # are NOT of interest.
    name_bin = LabelBinarizer()
    name_bin.fit(forest_sounds.mid.values.tolist() + ['None'])

    # Binarize labels for dataset
    # There are multiple labels per row, so we have to split them and do a binarization
    # on each column.  We then aggregate all the one-hot variables to get our final encoding.
    train_labels_split = train.positive_labels.str.split(',', expand=True)
    train_labels_split.fillna('None', inplace=True)
    train_label_binarized = name_bin.transform(train_labels_split[train_labels_split.columns[0]])
    for column in train_labels_split.columns:
        train_label_binarized |= name_bin.transform(train_labels_split[column])
    train_label_binarized = pd.DataFrame(train_label_binarized, columns=name_bin.classes_)
    del train_label_binarized['None']


    # Remove rows for uninteresting sounds
    train = train.loc[train_label_binarized.sum(axis=1) > 0]
    train.index = list(range(train.shape[0]))
    train_label_binarized = train_label_binarized.loc[train_label_binarized.sum(axis=1) > 0]
    train_label_binarized.index = list(range(train_label_binarized.shape[0]))

    # Translate mid to display name
    new_column_names = []
    for column in train_label_binarized.columns:
        new_column_names += [label_names.loc[label_names.mid == column].iloc[0].display_name]
    train_label_binarized.columns = new_column_names

    return train, train_label_binarized

############################################################################################
            # Downloads the youtube audio files
############################################################################################
def download_clip(YTID, start_seconds, end_seconds, target_path):
    """
    Downloads the youtube audio files
    """
    url = "https://www.youtube.com/watch?v=" + YTID

    #set the target path to download the audio files
    target_file = target_path + YTID + '-' + str(start_seconds) + '-' + str(end_seconds)+".wav"

    # No need to download audio file that already has been downloaded
    if os.path.isfile(target_file):
        return

    tmp_filename = target_path+ '/tmp_clip_' + YTID + '-' + str(start_seconds) + '-' + str(end_seconds)
    tmp_filename_w_extension = tmp_filename +'.wav'

    try:
        check_call(['youtube-dl', url,
                    '--audio-format', 'wav',
                    '-x', '-o', tmp_filename +'.%(ext)s'])
    except CalledProcessError:

        # do nothing
        print("Exception CalledProcessError!")
        return

    try:
        aud_seg = AudioSegment.from_wav(tmp_filename_w_extension)[start_seconds*1000:end_seconds*1000]

        aud_seg.export(target_file, format="wav")
    except:
        print("Error while reading file!")

    os.remove(tmp_filename_w_extension)




############################################################################################
                # To download the audio files from youtube
############################################################################################
def download_data(target_sounds_list, target_path):
    """
    Get the data necessary for downloading audio files
    """
    df, labels_binarized = get_csv_data(target_sounds_list)
    labels_csv = pd.read_csv("data/audioset/class_labels_indices.csv")

    # create a column with name of the labels given to sounds and also column with wav file names
    df['positive_labels'] = df['positive_labels'].apply(lambda arr: arr.split(','))
    df['labels_name'] = df['positive_labels'].map(lambda arr: np.concatenate([labels_csv.loc[labels_csv['mid'] == x].display_name for x in arr]))
    df['wav_file'] = df['YTID'].astype(str) + '-' + df['start_seconds'].astype(str) +\
                     '-' + df['end_seconds'].astype(str)+'.wav'

    #save the data frame which can be used for further balancing the data
    #and generating the embeddings for audio files.
    with open(RESULT.target_sounds+'_downloaded_base_dataframe.pkl', 'wb') as file_obj:
        pickle.dump(df, file_obj)

    print('Base dataframe is saved as " downloaded_base_dataframe.pkl "..!!')

    # create a path if it doesn't exists
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    threads = []

    for index in range(df.shape[0]):
        row = df.iloc[index]
        print("Downloading", str(index), "of", df.shape[0], ":", row.YTID)
        # download_clip(row.YTID, row.start_seconds, row.end_seconds)
        t = threading.Thread(target=download_clip,
                             args=(row.YTID, row.start_seconds, row.end_seconds, target_path))
        threads.append(t)
        t.start()
        if len(threads) > 4:
            threads[0].join()
            threads = threads[1:]
    for t in threads:
        t.join()




#call the function to dowload the target or sounds of Interest. Set the target
if __name__ == '__main__':
    download_data(SOUNDS_DICT[RESULT.target_sounds], RESULT.target_path)
