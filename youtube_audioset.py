"""
Reading all the data files and also Downloads audio files from youtube
"""
import json
from subprocess import check_call, CalledProcessError
import threading
import pickle
import os
import sys
import numpy as np
import pandas as pd
from pydub import AudioSegment
from sklearn.preprocessing import LabelBinarizer
import tensorflow.compat.v1 as tf


########################################################################
# Define the class and sublabels in it
########################################################################
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

###############################################################################
# Defining Ambient and Impact sounds as to what sounds it must comprise of.
###############################################################################
AMBIENT_SOUNDS = NATURE_SOUNDS
IMPACT_SOUNDS = EXPLOSION_SOUNDS + WOOD_SOUNDS + MOTOR_SOUNDS + \
    HUMAN_SOUNDS + TOOLS_SOUNDS + DOMESTIC_SOUNDS


###############################################################################
# function to get the dataframe from csv data
###############################################################################
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


###############################################################################
# Downloads the youtube audio files
###############################################################################
def download_clip(YTID, start_seconds, end_seconds, target_path):
    """
    Downloads the youtube audio files
    """
    url = "https://www.youtube.com/watch?v=" + YTID

    # set the target path to download the audio files
    target_file = target_path + YTID + '-' + str(start_seconds) + '-' + str(end_seconds) + ".wav"

    # No need to download audio file that already has been downloaded
    if os.path.isfile(target_file):
        return

    tmp_filename = target_path + '/tmp_clip_' + YTID + '-' + str(start_seconds) + '-' + str(end_seconds)
    tmp_filename_w_extension = tmp_filename + '.wav'

    try:
        check_call(['youtube-dl', url,
                    '--audio-format', 'wav',
                    '-x', '-o', tmp_filename + '.%(ext)s'])
    except CalledProcessError:

        # do nothing
        print("Exception CalledProcessError!")
        return

    try:
        aud_seg = AudioSegment.from_wav(tmp_filename_w_extension)[start_seconds * 1000:end_seconds * 1000]

        aud_seg.export(target_file, format="wav")
    except:
        print("Error while reading file!")

    os.remove(tmp_filename_w_extension)


###############################################################################
# To download the audio files from youtube
###############################################################################
def download_data(aggregate_name, target_sounds_list, target_path, file_limit):
    """
    Get the data necessary for downloading audio files
    """
    df, labels_binarized = get_csv_data(target_sounds_list)
    labels_csv = pd.read_csv("data/audioset/class_labels_indices.csv")

    assert(file_limit > 0)
    if file_limit < df.shape[0]:
        df = df.iloc[:file_limit]
        labels_binarized = labels_binarized.iloc[:file_limit]

    # create a column with name of the labels given to sounds and also column with wav file names
    df['positive_labels'] = df['positive_labels'].apply(lambda arr: arr.split(','))
    df['labels_name'] = df['positive_labels'].map(lambda arr: np.concatenate([labels_csv.loc[labels_csv['mid'] == x].display_name for x in arr]))
    df['wav_file'] = df['YTID'].astype(str) + '-' + df['start_seconds'].astype(str) +\
        '-' + df['end_seconds'].astype(str) + '.wav'

    # save the data frame which can be used for further balancing the data
    # and generating the embeddings for audio files.
    with open(aggregate_name + '_downloaded_base_dataframe.pkl', 'wb') as file_obj:
        pickle.dump(df, file_obj)

    print("Base dataframe is saved as", aggregate_name, "downloaded_base_dataframe.pkl ..!!")

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


###############################################################################
# slightly modified from
# https://stackoverflow.com/questions/42703849/audioset-and-tensorflow-understanding
###############################################################################
def read_audio_record(audio_record, output_to_file=None):
    """
    # https://stackoverflow.com/questions/42703849/audioset-and-tensorflow-understanding
    """
    vid_ids = []
    labels = []
    start_time_seconds = []  # in secondes
    end_time_seconds = []
    feat_audio = []
    count = 0

    # Workaround for tensorflow v1
    tf.disable_eager_execution()

    with tf.device("/cpu:0"):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.0001
        sess = tf.InteractiveSession(config=config)

        for example in tf.python_io.tf_record_iterator(audio_record):
            tf_example = tf.train.Example.FromString(example)
            vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
            labels.append(list(tf_example.features.feature['labels'].int64_list.value))
            start_time_seconds.append(tf_example.features.feature['start_time_seconds'].float_list.value[0])
            end_time_seconds.append(tf_example.features.feature['end_time_seconds'].float_list.value[0])
            tf_seq_example = tf.train.SequenceExample.FromString(example)
            n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
            audio_frame = []
            # iterate through frames
            for i in range(n_frames):
                audio_frame.append(tf.cast(tf.decode_raw(
                    tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],
                    tf.uint8),
                    tf.float32).eval())

            feat_audio.append([])
            feat_audio[count].append(audio_frame)
            count += 1
        sess.close()

    df = pd.DataFrame(list(zip(vid_ids, labels, start_time_seconds, end_time_seconds, feat_audio)),
                      columns=['video_id',
                               'labels',
                               'start_time_seconds',
                               'end_time_seconds',
                               'features'])
    df['features'] = df.features.apply(lambda x: np.array(x[0]))
    if output_to_file:
        df.to_pickle(audio_record.replace('.tfrecord', '.pkl'))
    return df


###############################################################################
# Read the recursive names from JSON file
###############################################################################
def get_recursive_sound_names(designated_sound_names, path_to_ontology, ontology_extension_paths=[]):
    """
    Read the recursive names from JSON file
    """
    if path_to_ontology is None:
        path_to_ontology = "externals/audioset_ontology/ontology.json"
    else:
        path_to_ontology = path_to_ontology + "externals/audioset_ontology/ontology.json"

    ontology_entries = json.load(open(path_to_ontology, 'r'))

    # Create two lookup tables (one with name as key and another with id as key)
    ontology_dict_from_id = {}
    ontology_dict_from_name = {}
    for entry in ontology_entries:
        ontology_dict_from_name[entry['name']] = entry
        ontology_dict_from_id[entry['id']] = entry

    del ontology_entries

    # Collect extensions and amend the ontology
    for ontology_extension_path in ontology_extension_paths:
        ontology_entries = json.load(open(ontology_extension_path, 'r'))

        for entry in ontology_entries:
            # If applicable, attach this node to the audioset ontology tree
            if entry['audioset_parent_id'] is not None:
                ontology_dict_from_id[entry['audioset_parent_id']]['child_ids'] += [entry['id']]

            ontology_dict_from_name[entry['name']] = entry
            ontology_dict_from_id[entry['id']] = entry

        del ontology_entries

    designated_sound_ids = [ontology_dict_from_name[sound]['id'] for sound in designated_sound_names]

    def get_ids(id):
        """
        Get the ID of the sounds
        """
        id_list = [id]
        for child_id in ontology_dict_from_id[id]['child_ids']:
            id_list += get_ids(child_id)
        return id_list

    recursive_sound_ids = []
    for id in designated_sound_ids:
        recursive_sound_ids += get_ids(id)
    recursive_sound_ids = list(set(recursive_sound_ids))

    recursive_sound_names = [ontology_dict_from_id[sound_id]['name'].lower() for sound_id in recursive_sound_ids]

    # Every sound name should be unique
    recursive_sound_names = set(recursive_sound_names)

    return recursive_sound_names


def get_all_sound_names(path_to_ontology):
    """
    Get all the sound names
    """
    return get_recursive_sound_names(AMBIENT_SOUNDS, path_to_ontology), get_recursive_sound_names(IMPACT_SOUNDS, path_to_ontology)


###############################################################################
# Read the tf.record files data
###############################################################################
def get_data():
    """
    Read the tf.record files data
    """
    label_names = pd.read_csv("data/audioset/class_labels_indices.csv")
    audio_files = ['bal_train/' + x for x in os.listdir('data/audioset/audioset_v1_embeddings/bal_train')] + \
                  ['unbal_train/' + x for x in os.listdir('data/audioset/audioset_v1_embeddings/unbal_train')]
    audio_files = [x for x in audio_files if x.endswith('.tfrecord')]
    audio_files.sort()
    path_prefix = 'data/audioset/audioset_v1_embeddings/'
    for audio_record in audio_files:
        if os.path.isfile(path_prefix + audio_record.replace('.tfrecord', '.pkl')):
            continue
        print("Reading", audio_record, "...")
        try:
            pid = os.fork()
        except OSError:
            sys.stderr.write("Could not create a child process\n")
            continue
        if pid == 0:
            read_audio_record(path_prefix + audio_record, True)
            os._exit(0)
        else:
            os.waitpid(pid, 0)
    pickle_files = ['bal_train/' + x for x in os.listdir('data/audioset/audioset_v1_embeddings/bal_train')] + \
        ['unbal_train/' + x for x in os.listdir('data/audioset/audioset_v1_embeddings/unbal_train')]
    pickle_files = [x for x in pickle_files if x.endswith('.pkl')]
    pickle_files.sort()

    ambient_sounds, impact_sounds = get_all_sound_names()
    print("Reading pickles...")
    df = []
    for audio_record in pickle_files:
        print("Reading", audio_record, "pickle")
        sub_df = pd.read_pickle(path_prefix + audio_record)

        def check_sounds(x):
            for sound in impact_sounds + ambient_sounds:
                if sound in x:
                    return True
            return False
        sub_df['labels'] = sub_df['labels'].map(lambda arr: [label_names.iloc[x].display_name for x in arr])
        sub_df = sub_df.loc[sub_df['labels'].map(check_sounds)]
        df += [sub_df]
    df = pd.concat(df, ignore_index=True)

    # Binarize the labels
    name_bin = LabelBinarizer().fit(ambient_sounds + impact_sounds)
    labels_split = df['labels'].apply(pd.Series).fillna('None')
    labels_binarized = name_bin.transform(labels_split[labels_split.columns[0]])
    for column in labels_split.columns:
        labels_binarized |= name_bin.transform(labels_split[column])
    labels_binarized = pd.DataFrame(labels_binarized, columns=name_bin.classes_)

    return df, labels_binarized
