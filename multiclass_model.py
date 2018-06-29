import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_recall_curve , confusion_matrix , accuracy_score, roc_auc_score, classification_report, hamming_loss
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
from ipywidgets import IntProgress
from sklearn.utils import shuffle
from mlxtend.plotting import plot_learning_curves, plot_decision_regions
from mlxtend.plotting import plot_confusion_matrix
from math import log
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, AveragePooling1D
from keras.layers.core import Lambda
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop

from keras_tqdm import TQDMNotebookCallback

from youtube_audioset import get_data, get_recursive_sound_names, get_all_sound_names
from youtube_audioset import explosion_sounds, motor_sounds, wood_sounds, human_sounds, nature_sounds, Wild_animals



#get the sound names
ambient_sounds, impact_sounds = get_all_sound_names()

explosion_sounds = get_recursive_sound_names(explosion_sounds)
motor_sounds = get_recursive_sound_names(motor_sounds)
wood_sounds = get_recursive_sound_names(wood_sounds)
human_sounds = get_recursive_sound_names(human_sounds)
nature_sounds = get_recursive_sound_names(nature_sounds)
#wild_animals=get_recursive_sound_names(Wild_animals)

#Read in the balanced data created by the balancing_dataset.py
with open('<path to balanced data>','rb') as f:
    df=pickle.load(f)
print(df.shape)
df['labels']=df['labels_name']

#List all the classes do be detected
all_sounds=['Motor_sound','Human_sound','Explosion_sound', 'Wood_sound']
all_sounds_list=explosion_sounds + motor_sounds + wood_sounds +human_sounds + nature_sounds

#Create a new labels column having classes that are to detected by mapping to repectively
df['labels_new']=df['labels_name'].apply(lambda arr: [ 'Motor_sound' if x  in motor_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Explosion_sound' if x  in explosion_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Nature_sound' if x  in nature_sounds[:-2] else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Human_sound' if x  in human_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Wood_sound' if x  in wood_sounds else x for x in arr])


#Binarize the labels
name_bin = MultiLabelBinarizer().fit(df['labels_new'])
labels_binarized = name_bin.transform(df['labels_new'])
labels_binarized_all = pd.DataFrame(labels_binarized, columns = name_bin.classes_)
labels_binarized=labels_binarized_all[all_sounds]


print df.shape[0], "examples"
print(labels_binarized.mean())


df_filtered = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]
# df_filtered = df.loc[df['labels'].apply(lambda x: (len(x) == 1)) & df.features.apply(lambda x: x.shape[0] == 10)]
labels_filtered = labels_binarized.loc[df_filtered.index,:]
#print(labels_filtered)

#split data to get train and test data
df_train, df_test, labels_binarized_train, labels_binarized_test = train_test_split(df_filtered, labels_filtered,
                                                                      test_size=0.33, random_state=42)

X_train = np.array(df_train.features.apply(lambda x: x.flatten()).tolist())
X_train_standardized = X_train / 255
X_test = np.array(df_test.features.apply(lambda x: x.flatten()).tolist())
X_test_standardized = X_test / 255


def create_keras_model():
    # create model
    model = Sequential()
    model.add(Conv1D(40, input_shape=(1280,1), kernel_size=128, strides=128, activation='relu', padding='same'))
    model.add(Conv1D(100, kernel_size=3, activation='relu', padding='same'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Lambda(lambda x:1-x, output_shape=(10,5)))
    model.add(MaxPooling1D(10))
    model.add(Lambda(lambda x:1-x, output_shape=(1,5)))
    model.add(Flatten())
    print model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=5e-4), metrics=['accuracy'])
    return model
#
clf2_train = X_train.reshape((-1,1280,1))
clf2_test = X_test.reshape((-1,1280,1))
clf2_train_target = labels_binarized_train
clf2_test_target = labels_binarized_test

#Implementing the train_on_batch techinque to solve the memory issue

model = create_keras_model()
nb_epochs=25
minibatch_size=100
for epoch in range(nb_epochs):
    print('Epoch ' + str(epoch) +' out of '+ str(nb_epochs) )


    #Randomize the data point
    ex_train , ey_train_target = shuffle(clf2_train, clf2_train_target)

    for i in range(0, clf2_train.shape[0],minibatch_size):

        clf2_train_mini= ex_train[i:i + minibatch_size]
        clf2_train_target_mini = ey_train_target[i:i + minibatch_size]

        model.train_on_batch(clf2_train_mini,clf2_train_target_mini, class_weight={0:1,1:1,2:1,3:3})


#Make the preditions on test and train data
clf2_train_prediction = model.predict(clf2_train).ravel().round()
clf2_train_prediction_prob = model.predict(clf2_train).ravel()
clf2_test_prediction = model.predict(clf2_test).round()
clf2_test_prediction_prob = model.predict(clf2_test).ravel()


#Print out the confusion_matrix and classification_report
print(clf2_test_target.values.argmax(axis=1).shape)
print('        Confusion Matrix          ')
print('============================================')
result = confusion_matrix(clf2_test_target.values.argmax(axis=1), clf2_test_prediction.argmax(axis=1))
print(result)
print('                 Classification Report      ')
print('============================================')
cl_report=classification_report(clf2_test_target.values.argmax(axis=1), clf2_test_prediction.argmax(axis=1))
print(cl_report)
accuracy=accuracy_score(clf2_test_target.values.argmax(axis=1), clf2_test_prediction.argmax(axis=1))
print 'Accuracy :', accuracy
HL = hamming_loss(clf2_test_target.values.argmax(axis=1), clf2_test_prediction.argmax(axis=1))
print 'Hamming Loss :',HL

# Save your model
# model.save_weights('weights.model')
# print('Weights_saved')'''
