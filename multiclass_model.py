#Import the necessary functions and libraries
import pandas as pd
import numpy as np
from scipy import spatial
import pickle
from ipywidgets import IntProgress
import matplotlib.pyplot as plt
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
from sklearn.utils import shuffle
from mlxtend.plotting import plot_learning_curves, plot_decision_regions
from mlxtend.plotting import plot_confusion_matrix
from math import log
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, AveragePooling1D
from keras.layers.core import Lambda
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop, Adam
from keras_tqdm import TQDMNotebookCallback
from youtube_audioset import get_data, get_recursive_sound_names, get_all_sound_names
from youtube_audioset import explosion_sounds, motor_sounds, wood_sounds, human_sounds, nature_sounds, Wild_animals,domestic_sounds, tools
import balanced_data_priority_2


# Get all sound names
ambient_sounds, impact_sounds = get_all_sound_names()
explosion_sounds = get_recursive_sound_names(explosion_sounds)
motor_sounds = get_recursive_sound_names(motor_sounds)
wood_sounds = get_recursive_sound_names(wood_sounds)
human_sounds = get_recursive_sound_names(human_sounds)
nature_sounds = get_recursive_sound_names(nature_sounds)
domestic_sounds = get_recursive_sound_names(domestic_sounds)
tools = get_recursive_sound_names(tools)
#wild_animals=get_recursive_sound_names(Wild_animals)


#Importing balanced data from the function
df = balancing_dataset.balanced_data()
print df.shape[0], "examples"

# Different classes of sounds. You can increase the class by adding the necesssary sounds of that class which will be used for training
all_sounds=['Motor_sound','Explosion_sound','Human_sound','Nature_sound','Domestic_animals','Tools']
all_sounds_list=explosion_sounds + motor_sounds + human_sounds + nature_sounds



# Map all the sounds into their respective classes
df['labels_new']=df['labels_name'].apply(lambda arr: [ 'Motor_sound' if x  in motor_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Explosion_sound' if x  in explosion_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Nature_sound' if x  in nature_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Human_sound' if x  in human_sounds else x for x in arr])
# df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Wood_sound' if x  in wood_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Domestic_animals' if x  in domestic_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Tools' if x  in tools else x for x in arr])
# df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Wild_animals' if x  in Wild_animals else x for x in arr])


#Binarize the labels. Its a Multi-label binarizer
name_bin = MultiLabelBinarizer().fit(df['labels_new'])
labels_binarized = name_bin.transform(df['labels_new'])
labels_binarized_all = pd.DataFrame(labels_binarized, columns = name_bin.classes_)
labels_binarized=labels_binarized_all[all_sounds]


#print out the number and percenatge of each class examples
print(labels_binarized.mean())


#Filtering the sounds that are exactly 10 seconds
df_filtered = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]
labels_filtered = labels_binarized.loc[df_filtered.index,:]
#print(labels_filtered)


#split the data into train and test data
df_train, df_test, labels_binarized_train, labels_binarized_test = train_test_split(df_filtered, labels_filtered,
                                                                      test_size=0.33, random_state=42)

#preprecess the data into required structure
X_train = np.array(df_train.features.apply(lambda x: x.flatten()).tolist())
X_train_standardized = X_train / 255
X_test = np.array(df_test.features.apply(lambda x: x.flatten()).tolist())
X_test_standardized = X_test / 255


#create the keras model
def create_keras_model():
    # create model
    model = Sequential()
    model.add(Conv1D(500, input_shape=(1280,1), kernel_size=128, strides=128, activation='relu', padding='same'))
    model.add(Dense(500,activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(MaxPooling1D(10))
    model.add(Dense(6, activation='sigmoid'))
    model.add(Flatten())
    print model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4,epsilon=1e-8), metrics=['accuracy'])
    return model


#reshaping the train and test data so as to align with input for model
clf2_train = X_train.reshape((-1,1280,1))
clf2_test = X_test.reshape((-1,1280,1))
clf2_train_target = labels_binarized_train
clf2_test_target = labels_binarized_test


#Implementing using the keras usual training techinque
model=create_keras_model()
model_traing=model.fit(clf2_train, clf2_train_target,
          epochs=50, batch_size=100, verbose=1,
          validation_data = (clf2_test, clf2_test_target))


#Predict on train and test data
clf2_train_prediction = model.predict(clf2_train).round()
clf2_train_prediction_prob = model.predict(clf2_train)
clf2_test_prediction = model.predict(clf2_test).round()
clf2_test_prediction_prob = model.predict(clf2_test)


# To get the Misclassified examples
df_test['actual_labels']=np.split(labels_binarized_test.values ,df_test.shape[0])
df_test['predicted_labels']=np.split(clf2_test_prediction,df_test.shape[0])
df_test['predicted_prob']=np.split(clf2_test_prediction_prob,df_test.shape[0])
misclassified_array = clf2_test_prediction != clf2_test_target
misclassified_examples = np.any(misclassified_array, axis=1)


# print misclassified number of examples
print 'Misclassified number of examples :' ,df_test[misclassified_examples].shape[0]


#Print confusion matrix and classification_report
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

#Save the model weights
# model.save_weights('multiclass_weights.model')
# print('Weights_saved')
