from mlxtend.plotting import plot_learning_curves, plot_decision_regions
from mlxtend.plotting import plot_confusion_matrix
from math import log
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, AveragePooling1D, TimeDistributed, MaxPooling2D
from keras.layers.core import Lambda
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop, Adam
from keras_tqdm import TQDMNotebookCallback
from youtube_audioset import get_data, get_recursive_sound_names, get_all_sound_names
from youtube_audioset import explosion_sounds, motor_sounds, wood_sounds, human_sounds, nature_sounds, Wild_animals,domestic_sounds
import balanced_data_priority_2
import math
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import math
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
import ast
import pylab
import pydub
import glob
import pickle
import scipy.signal
from sklearn import metrics
from keras.layers.core import Dropout
from pydub import AudioSegment
from matplotlib import pyplot as plt
import scipy.io.wavfile
import balanced_data_priority_2
import _copy_goertzel
import arbitary
import balance_data_priority


#give the path for saved frequency component files
Path_of_saved_freq_components = '/media/wildly/1TB-HDD/goertzel_data_8k_resampled_800_1600_2000_2300/'


#get the data of each sounds seperately and then concat all sounds to get balanced data
mot,hum,wod,exp,dom,tools,wild,nat = frequency_component_files.get_req_sounds(Path_of_saved_freq_components)

#try to balance the number of examples. Here we need to balance as Impact Vs Ambient , but not as maultilabel sounds
df = pd.concat ( [mot[:3000],hum[:1700],wod[:500],exp[:1200],dom[:1100],tools[:1500],wild[:1000],nat[:8000]] ,ignore_index=True )


# execute the labels binarized by importing the youtube_audioset function
from youtube_audioset import get_data, get_recursive_sound_names, get_all_sound_names
from youtube_audioset import explosion_sounds, motor_sounds, wood_sounds, human_sounds, nature_sounds, domestic_sounds
ambient_sounds, impact_sounds = get_all_sound_names()
explosion_sounds = get_recursive_sound_names(explosion_sounds)
motor_sounds = get_recursive_sound_names(motor_sounds)
wood_sounds = get_recursive_sound_names(wood_sounds)
human_sounds = get_recursive_sound_names(human_sounds)
nature_sounds = get_recursive_sound_names(nature_sounds)
domestic_sounds = get_recursive_sound_names(domestic_sounds)


#Binarize the labels
name_bin = LabelBinarizer().fit(ambient_sounds + impact_sounds)
labels_split = df['labels_name'].apply(pd.Series).fillna('None')
labels_binarized = name_bin.transform(labels_split[labels_split.columns[0]])
for column in labels_split.columns:
    labels_binarized |= name_bin.transform(labels_split[column])
labels_binarized = pd.DataFrame(labels_binarized, columns = name_bin.classes_)

df, labels_binarized = shuffle(df , labels_binarized,random_state=20)

print 'Binarized labels shape :', labels_binarized.shape
print "Percentage Impact Sounds:", (labels_binarized[impact_sounds].sum(axis=1) > 0).mean()
print "Percentage Ambient Sounds:", (labels_binarized[ambient_sounds].sum(axis=1) > 0).mean()


# split up the data
df_train, df_test, labels_binarized_train, labels_binarized_test = train_test_split(df, labels_binarized,
                                                                      test_size=0.1, random_state=42
)

#create the time distributed model
def create_keras_model():
    # create mode
    model = Sequential()
    model.add(TimeDistributed(Conv1D(100,kernel_size=200,strides=100,activation='relu', padding='same'), input_shape=(10,8000,4)))
    model.add(TimeDistributed(Conv1D(100, kernel_size=4, activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling1D(80)))
    model.add(TimeDistributed(Dense(60, activation='relu')))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.add(MaxPooling2D((10,1)))
    model.add(Flatten())
    print model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(1e-3), metrics=['accuracy'])
    return model

# define the target labels for test and train
clf1_test_target = labels_binarized_test.loc[:,impact_sounds].any(axis=1)
clf1_train_target_mini = np.array(labels_binarized_train.loc[:,impact_sounds].any(axis=1),dtype=float)

#Train the model
model = create_keras_model()
clf1_train_mini= []
clf1_test=[]
print 'Reading Test files ..!'

# Read all the test data first
for wav_file in df_test['wav_file'].tolist():
    try:
        with open( Path_of_saved_freq_components + wav_file[:11]+'.pkl','rb') as f:
            arb_wav = pickle.load(f)
        clf1_test.append(np.dstack( (arb_wav[0].reshape((10,8000)) ,arb_wav[1].reshape((10,8000)), arb_wav[2].reshape((10,8000)), arb_wav[3].reshape((10,8000)) ) ))

        #except any error then remove that file manually and then run the process again
    except :
        print 'Test Pickling Error'
        print wav_file

#reshaping test data and applying normalization
print np.array(clf1_test).shape
clf1_test = np.array(clf1_test).reshape((-1,10,8000,4))
clf1_test = clf1_test / np.linalg.norm(clf1_test)

print "Reading Training files..!!"
for wav in df_train['wav_file'].tolist():
    try:
        with open( Path_of_saved_freq_components + wav[:11]+'.pkl','rb') as f:
            arb_wav = pickle.load(f)
        clf1_train_mini.append( np.dstack( (arb_wav[0].reshape((10,8000)) ,arb_wav[1].reshape((10,8000)), arb_wav[2].reshape((10,8000)), arb_wav[3].reshape((10,8000)) ) ))
    except:
        print 'Train pickling Error '
        print wav

# reshaping the traininig data and applying normalization
clf1_train_mini = np.array(clf1_train_mini).reshape((-1,10,8000,4))
clf1_train_mini = clf1_train_mini/np.linalg.norm(clf1_train_mini)

#start training on model
model.fit( clf1_train_mini, clf1_train_target_mini, epochs=30, verbose=1, validation_data=(clf1_test,clf1_test_target) )


#predict out of the model

clf1_train_prediction_prob = model.predict(clf1_train_mini).ravel()
clf1_train_prediction = model.predict(clf1_train_mini).ravel().round()
# clf1_train_prediction = np.array([0 if i<0.45 else 1 for i in clf1_train_prediction_prob])

clf1_test_prediction_prob = model.predict(clf1_test).ravel()
clf1_test_prediction = model.predict(clf1_test).ravel().round()
# clf1_test_prediction = np.array([0 if i<0.22 else 1 for i in clf1_test_prediction_prob])
print clf1_test_prediction_prob[clf1_test_prediction_prob<0.3]

# print train and test acuuracy
print "Train Accuracy:", (clf1_train_prediction == clf1_train_target_mini).mean()
print "Test Accuracy:", (clf1_test_prediction == clf1_test_target).mean()

#print out the confusion matrix for train data
clf1_conf_train_mat = pd.crosstab(clf1_train_target_mini, clf1_train_prediction, margins=True)
print("Training Precision and recall for Keras model")
print('=============================================')
print "Train Precision:", clf1_conf_train_mat[True][True] / float(clf1_conf_train_mat[True]['All'])
print "Train Recall:", clf1_conf_train_mat[True][True] / float(clf1_conf_train_mat['All'][True])
print "Train Accuracy:", (clf1_train_prediction == clf1_train_target_mini).mean()
print(clf1_conf_train_mat)


#print out the confusion matrix for test data
clf1_conf_test_mat = pd.crosstab(clf1_test_target, clf1_test_prediction, margins=True)
print("Testing Precision and recall for Keras model")
print('=============================================')
print "Test Precision:", clf1_conf_test_mat[True][True] / float(clf1_conf_test_mat[True]['All'])
print "Test Recall:", clf1_conf_test_mat[True][True] / float(clf1_conf_test_mat['All'][True])
print "Test Accuracy:", (clf1_test_prediction == clf1_test_target).mean()
print(clf1_conf_test_mat)

# calculate the f1 score and print it out
f_score = metrics.f1_score(clf1_test_target, clf1_test_prediction)
print 'F1 score is  : ', f_score

#save the model
model.save_weights('Goertzel_model_8k_weights_time.h5')
