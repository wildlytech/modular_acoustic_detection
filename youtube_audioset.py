import pandas as pd
from pydub import AudioSegment
from subprocess import call, check_call, CalledProcessError
import threading

import os

from sklearn.preprocessing import LabelBinarizer

ambient_sounds = [
"Silence",
"Stream",
"Wind noise (microphone)",
"Wind",
"Rustling leaves",
"Howl",
"Raindrop",
"Rain on surface",
"Rain"
]

impact_sounds = [
"Fire",
"Fireworks",
"Burst, pop",
"Eruption",
"Crackle",
"Thunderstorm",
"Gunshot, gunfire",
"Explosion",
"Boom",
"Chainsaw",
"Wood",
"Medium engine (mid frequency)",
"Light engine (high frequency)",
"Heavy engine (low frequency)",
"Engine starting",
"Engine",
"Motor vehicle (road)",
"Vehicle",
"Chatter",
"Conversation"
]

def get_data():
    label_names = pd.read_csv("data/audioset/class_labels_indices.csv")

    balanced_train = pd.read_csv("data/audioset/balanced_train_segments.csv",quotechar='"', skipinitialspace=True, skiprows=2)
    balanced_train.columns = [balanced_train.columns[0][2:]] + balanced_train.columns[1:].values.tolist()

    unbalanced_train = pd.read_csv("data/audioset/unbalanced_train_segments.csv",quotechar='"', skipinitialspace=True, skiprows=2)
    unbalanced_train.columns = [unbalanced_train.columns[0][2:]] + unbalanced_train.columns[1:].values.tolist()

    train = pd.concat([unbalanced_train, balanced_train], axis=0, ignore_index=True)

    # Filter out sounds we don't care about
    forest_sounds = pd.Series(ambient_sounds + impact_sounds, name='display_name')
    forest_sounds = pd.DataFrame(forest_sounds).merge(label_names, how='inner', left_on='display_name', right_on='display_name')

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
    train_label_binarized = pd.DataFrame(train_label_binarized, columns = name_bin.classes_)
    del train_label_binarized['None']

    # Remove rows for uninteresting sounds
    train = train.loc[train_label_binarized.sum(axis=1) > 0]
    train.index = range(train.shape[0])
    train_label_binarized = train_label_binarized.loc[train_label_binarized.sum(axis=1) > 0]
    train_label_binarized.index = range(train_label_binarized.shape[0])

    # Translate mid to display name
    new_column_names = []
    for column in train_label_binarized.columns:
        new_column_names += [label_names.loc[label_names.mid == column].iloc[0].display_name]
    train_label_binarized.columns = new_column_names

    return train, train_label_binarized

def download_clip(YTID, start_seconds, end_seconds):
    url = "https://www.youtube.com/watch?v=" + YTID
    target_file = 'sounds/audioset/' + YTID + '-' + str(start_seconds) + '-' + str(end_seconds)+".wav"

    # No need to download audio file that already has been downloaded
    if os.path.isfile(target_file):
        return

    tmp_filename = 'sounds/tmp_clip_' + YTID + '-' + str(start_seconds) + '-' + str(end_seconds)
    tmp_filename_w_extension = tmp_filename +'.wav'

    try:
        check_call(['youtube-dl', url, '--audio-format', 'wav', '-x', '-o', tmp_filename +'.%(ext)s'])
    except CalledProcessError:
        # do nothing
        print "Exception CalledProcessError!"
        return

    try:
        aud_seg = AudioSegment.from_wav(tmp_filename_w_extension)[start_seconds*1000:end_seconds*1000]

        aud_seg.export(target_file, format="wav")
    except:
        print "Error while reading file!"

    os.remove(tmp_filename_w_extension)


def download_data():

    df, labels_binarized = get_data()

    if not os.path.exists('sounds'):
        os.makedirs('sounds')

    if not os.path.exists('sounds/audioset'):
        os.makedirs('sounds/audioset')

    threads = []

    for index in range(df.shape[0]):
        row = df.iloc[index]

        print "Downloading", str(index), "of", df.shape[0], ":", row.YTID

        # download_clip(row.YTID, row.start_seconds, row.end_seconds)
        t = threading.Thread(target=download_clip, args=(row.YTID, row.start_seconds, row.end_seconds))
        threads.append(t)
        t.start()

        if len(threads) > 4:
            threads[0].join()
            threads = threads[1:]

    for t in threads:
        t.join()
