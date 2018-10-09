import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
from ipywidgets import IntProgress

from mlxtend.plotting import plot_learning_curves, plot_decision_regions
from mlxtend.plotting import plot_confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop

from keras_tqdm import TQDMNotebookCallback

from youtube_audioset import get_data, get_recursive_sound_names, get_all_sound_names
from youtube_audioset import explosion_sounds, motor_sounds, wood_sounds, human_sounds, nature_sounds


ambient_sounds, impact_sounds = get_all_sound_names()

explosion_sounds = get_recursive_sound_names(explosion_sounds)
motor_sounds = get_recursive_sound_names(motor_sounds)
wood_sounds = get_recursive_sound_names(wood_sounds)
human_sounds = get_recursive_sound_names(human_sounds)
nature_sounds = get_recursive_sound_names(nature_sounds)

#Read the balanced data created by running the balancing_datasets.py

#Note that this is binary classification. Balancing must be  [ Ambient ] vs  [ Impact ]
df = balancing_dataset.balanced_data()

# Binarize the labels
name_bin = LabelBinarizer().fit(ambient_sounds + impact_sounds)
labels_split = df['labels_name'].apply(pd.Series).fillna('None')
labels_binarized = name_bin.transform(labels_split[labels_split.columns[0]])
for column in labels_split.columns:
    labels_binarized |= name_bin.transform(labels_split[column])
labels_binarized = pd.DataFrame(labels_binarized, columns = name_bin.classes_)


print(labels_binarized.shape)
print df.shape[0], "examples"

# print the percentage of Impact and Ambinet sounds
print "Percentage Impact Sounds:", (labels_binarized[impact_sounds].sum(axis=1) > 0).mean()
print "Percentage Ambient Sounds:", (labels_binarized[ambient_sounds].sum(axis=1) > 0).mean()

print(labels_binarized.mean())


#Filter out the sounds that are having 10 seconds duration.
df_filtered = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]
labels_filtered = labels_binarized.loc[df_filtered.index,:]


#split the data into train and test
df_train, df_test, labels_binarized_train, labels_binarized_test = train_test_split(df_filtered, labels_filtered,
                                                                      test_size=0.33, random_state=42,
                                                                      stratify=labels_filtered.any(axis=1)*1)

X_train = np.array(df_train.features.apply(lambda x: x.flatten()).tolist())
X_train_standardized = X_train / 255
X_test = np.array(df_test.features.apply(lambda x: x.flatten()).tolist())
X_test_standardized = X_test / 255

y_train = (labels_binarized_train[impact_sounds].any(axis=1)*1).values
y_test = (labels_binarized_test[impact_sounds].any(axis=1)*1).values


# Print the percentage of each sounds in whole data
print labels_filtered.loc[:,explosion_sounds].any(axis=1).mean()
print labels_filtered.loc[:,motor_sounds].any(axis=1).mean()
print labels_filtered.loc[:,wood_sounds].any(axis=1).mean()
print labels_filtered.loc[:,human_sounds].any(axis=1).mean()
print labels_filtered.loc[:,nature_sounds].any(axis=1).mean()


labels_filtered.loc[:,impact_sounds].any(axis=1).mean()


# Try experimenting with Logistic regression algorithm
clf1_ = LogisticRegression(max_iter=1000)
clf1_train = X_train
clf1_test = X_test

# Assign the Impact sounds( target sounds) as  1's  and ambient sounds as 0's
clf1_train_target = labels_binarized_train.loc[:,impact_sounds].any(axis=1)
clf1_test_target = labels_binarized_test.loc[:,impact_sounds].any(axis=1)

#fit the train data 
clf1_.fit(clf1_train, clf1_train_target)

# Predict on the trained model
clf1_train_prediction = clf1_.predict(clf1_train)
clf1_test_prediction = clf1_.predict(clf1_test)
clf1_test_prediction_prob = clf1_.predict_proba(clf1_test)[:,1]

# Print out the confusion matrix for Train data
clf1_conf_train_mat = pd.crosstab(clf1_train_target, clf1_train_prediction, margins=True)
print('Train precsion and recall for Logistic regression')
print('=============================================')


print "Train Precision:", clf1_conf_train_mat[True][True] / float(clf1_conf_train_mat[True]['All'])
print "Train Recall:", clf1_conf_train_mat[True][True] / float(clf1_conf_train_mat['All'][True])
print "Train Accuracy:", (clf1_train_prediction == clf1_train_target).mean()

print(clf1_conf_train_mat)

# Print out the confusion matrix for test data
clf1_conf_test_mat = pd.crosstab(clf1_test_target, clf1_test_prediction, margins=True)
print('Test precsion and recall for Logistic regression')
print('=============================================')


print "Test Precision:", clf1_conf_test_mat[True][True] / float(clf1_conf_test_mat[True]['All'])
print "Test Recall:", clf1_conf_test_mat[True][True] / float(clf1_conf_test_mat['All'][True])
print "Test Accuracy:", (clf1_test_prediction == clf1_test_target).mean()
print(clf1_conf_test_mat)


# create the keras neural netwrok model
def create_keras_model():
    # create model
    model = Sequential()
    model.add(Conv1D(40, input_shape=(1280,1), kernel_size=128, strides=128, activation='relu', padding='same'))
    model.add(Conv1D(100, kernel_size=3, activation='relu', padding='same'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.add(MaxPooling1D(10))
    model.add(Flatten())
    print model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=5e-4), metrics=['accuracy'])
    return model

# Assign the train and test data and reshape so as to align it to neural network model's input. 
clf2_train = X_train.reshape((-1,1280,1))
clf2_test = X_test.reshape((-1,1280,1))
clf2_train_target = labels_binarized_train.loc[:,impact_sounds].any(axis=1)
clf2_test_target = labels_binarized_test.loc[:,impact_sounds].any(axis=1)

# call the model and start training 
model = create_keras_model()
model_traing=model.fit(clf2_train, clf2_train_target,
          epochs=500, batch_size=5000, verbose=False,
          validation_data = (clf2_test, clf2_test_target))

# Predict on test and train data using the tranined weights
clf2_train_prediction = model.predict(clf2_train).ravel().round()
clf2_test_prediction = model.predict(clf2_test).ravel().round()
clf2_test_prediction_prob = model.predict(clf2_test).ravel()

# Accuracy of Train and test 
print "Train Accuracy:", (clf2_train_prediction == clf2_train_target).mean()
print "Test Accuracy:", (clf2_test_prediction == clf2_test_target).mean()

#print out the confusion matrix for train data
clf2_conf_train_mat = pd.crosstab(clf2_train_target, clf2_train_prediction, margins=True)
print("Training Precision and recall for Keras model")
print('=============================================')
print "Train Precision:", clf2_conf_train_mat[True][True] / float(clf2_conf_train_mat[True]['All'])
print "Train Recall:", clf2_conf_train_mat[True][True] / float(clf2_conf_train_mat['All'][True])
print "Train Accuracy:", (clf2_train_prediction == clf2_train_target).mean()

print(clf2_conf_train_mat)


#print out the confusion matrix for test data
clf2_conf_test_mat = pd.crosstab(clf2_test_target, clf2_test_prediction, margins=True)
print("Testing Precision and recall for Keras model")
print('=============================================')


print "Test Precision:", clf2_conf_test_mat[True][True] / float(clf2_conf_test_mat[True]['All'])
print "Test Recall:", clf2_conf_test_mat[True][True] / float(clf2_conf_test_mat['All'][True])
print "Test Accuracy:", (clf2_test_prediction == clf2_test_target).mean()

print(clf2_conf_test_mat)
