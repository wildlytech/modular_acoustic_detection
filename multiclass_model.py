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




ambient_sounds, impact_sounds = get_all_sound_names()

explosion_sounds = get_recursive_sound_names(explosion_sounds)
motor_sounds = get_recursive_sound_names(motor_sounds)
wood_sounds = get_recursive_sound_names(wood_sounds)
human_sounds = get_recursive_sound_names(human_sounds)
nature_sounds = get_recursive_sound_names(nature_sounds)
#wild_animals=get_recursive_sound_names(Wild_animals)


with open('new_df_req_153k_32k.pkl','rb') as f:
    df=pickle.load(f)
print(df.shape)
df['labels']=df['labels_name']
# print(df['labels'])
all_sounds=['Motor_sound','Human_sound','Explosion_sound', 'Wood_sound']
all_sounds_list=explosion_sounds + motor_sounds + wood_sounds +human_sounds + nature_sounds
#Binarize the labels

df['labels_new']=df['labels_name'].apply(lambda arr: [ 'Motor_sound' if x  in motor_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Explosion_sound' if x  in explosion_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Nature_sound' if x  in nature_sounds[:-2] else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Human_sound' if x  in human_sounds else x for x in arr])
df['labels_new']=df['labels_new'].apply(lambda arr: [ 'Wood_sound' if x  in wood_sounds else x for x in arr])
# print(df['labels_new'])

# print(df['labels_new'])

name_bin = MultiLabelBinarizer().fit(df['labels_new'])
#labels_split = df['labels_new'].apply(pd.Series).fillna('None')
#print(labels_split)
#print(name_bin.classes_)
labels_binarized = name_bin.transform(df['labels_new'])
# for column in labels_split.columns:
#     labels_binarized |= name_bin.transform(labels_split[column])
labels_binarized_all = pd.DataFrame(labels_binarized, columns = name_bin.classes_)
labels_binarized=labels_binarized_all[all_sounds]
#print(labels_binarized)



# #with open('new_labels_binarized_bal.pkl','rb') as f:
# #    labels_binarized=pickle.load(f)
# print(labels_binarized.shape)

print df.shape[0], "examples"

# print "Percentage Impact Sounds:", (labels_binarized['Motor_sound'].sum(axis=0) > 0).mean()
# print "Percentage Ambient Sounds:", (labels_binarized['Explosion_sound'].sum(axis=0) > 0).mean()

print(labels_binarized.mean())

# ex_pc=input('Number of examples per class:')
# df=pd.concat([df]*ex_pc, ignore_index=True)
# labels_binarized=pd.concat([labels_binarized]*ex_pc,ignore_index=True)
#
# print(df.shape[0])
#
# print "Percentage Impact Sounds:", (labels_binarized[impact_sounds].sum(axis=1) > 0).mean()
# print "Percentage Ambient Sounds:", (labels_binarized[ambient_sounds].sum(axis=1) > 0).mean()
#


df_filtered = df.loc[df.features.apply(lambda x: x.shape[0] == 10)]
# df_filtered = df.loc[df['labels'].apply(lambda x: (len(x) == 1)) & df.features.apply(lambda x: x.shape[0] == 10)]
labels_filtered = labels_binarized.loc[df_filtered.index,:]
#print(labels_filtered)

df_train, df_test, labels_binarized_train, labels_binarized_test = train_test_split(df_filtered, labels_filtered,
                                                                      test_size=0.33, random_state=42)

X_train = np.array(df_train.features.apply(lambda x: x.flatten()).tolist())
X_train_standardized = X_train / 255
X_test = np.array(df_test.features.apply(lambda x: x.flatten()).tolist())
X_test_standardized = X_test / 255
# clf2_train_target_1 = np.array(labels_binarized_train.stack().tolist())
# clf2_test_target_2 = np.array(labels_binarized_test.stack().tolist())
#
# y_train = (labels_binarized_train[impact_sounds].any(axis=1)*1).values
# labels_binarized_test=labels_binarized_test.loc[df_test.index,:]
# y_test = (labels_binarized_test[impact_sounds].any(axis=1)*1).values

# print labels_filtered.loc[:,'Explosion_sound'].any(axis=1).mean()
# print labels_filtered.loc[:,'Motor_sound'].any(axis=1).mean()
# print labels_filtered.loc[:,'Wood_sound'].any(axis=1).mean()
# print labels_filtered.loc[:,'Human_sound'].any(axis=1).mean()
# print labels_filtered.loc[:,'Nature_sound'].any(axis=1).mean()
# print labels_filtered.loc[:,'Wild_animal'].any(axis=1).mean()

# from keras import backend as K
# from keras.layers.core import Lambda
# from keras.layers.merge import Concatenate
#
# def min_max_pool2d(x):
#
#     min_x = -K.pool1d(-x, pool_size=10)
#     # if K.image_dim_ordering() == 'th':
#     #     channel = 1
#     # else:
#     #     channel = -1
#     # return Concatenate( min_x, axis=channel)  # concatenate on channel
#     return min_x
#
# def min_max_pool2d_output_shape(input_shape):
#     shape = list(input_shape)
#     if K.image_dim_ordering() == 'th':
#         shape[2] /= 10
#         shape[3] /= 10
#     else:
#         shape[3] *= 2
#         shape[1] /= 10
#         shape[2] /= 10
#     return tuple(shape)


# replace maxpooling layer

def create_keras_model():
    # create model
    model = Sequential()
    model.add(Conv1D(40, input_shape=(1280,1), kernel_size=128, strides=128, activation='relu', padding='same'))
    model.add(Conv1D(200, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(100, kernel_size=4, activation='relu', padding='same'))
    # model.add(Dense(100,activation='relu'))
    model.add(Dense(40, activation='relu'))
    # model.add(Dense(10,activation='relu'))
    model.add(Dense(4, activation='sigmoid'))
    model.add(Lambda(lambda x:1-x, output_shape=(10,4)))
    # model.add(MaxPooling1D(10))
    model.add(MaxPooling1D(10))
    model.add(Lambda(lambda x:1-x, output_shape=(1,4)))
    model.add(Flatten())
    print model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=5e-4), metrics=['accuracy'])
    return model
#
clf2_train = X_train.reshape((-1,1280,1))
clf2_test = X_test.reshape((-1,1280,1))
clf2_train_target = labels_binarized_train
#print(clf2_train_target)
clf2_test_target = labels_binarized_test
# print(clf2_train_target)
#
# print(clf2_train_target.shape)
# print(clf2_train.shape)

# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, clf2_train_target)
# knn_prediction=knn.predict(X_test)
# print(knn_prediction)


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


# model_traing=model.fit(clf2_train, clf2_train_target,
#           epochs=100, batch_size=500, verbose=False,
#           validation_data = (clf2_test, clf2_test_target))


clf2_train_prediction = model.predict(clf2_train).ravel().round()
clf2_train_prediction_prob = model.predict(clf2_train).ravel()
clf2_test_prediction = model.predict(clf2_test).round()
# print(clf2_test_prediction.shape)
# print(clf2_test_target.shape)

clf2_test_prediction_prob = model.predict(clf2_test).ravel()
# print(df_test.shape)
# print(clf2_test_prediction_prob[:,1])
# lst=clf2_test_prediction_prob.tolist()
#
# Decision_kr=[]
# for i in range(len(lst)):
#     Decision_kr.append('%.3f' %lst[i])
# df_test['Decision_kr']=Decision_kr
# # print(clf2_test_target.shape)
# # print(clf2_test_prediction.shape)
# print(len(labels_binarized_test.values.tolist()))
# actual=labels_binarized_test.values.tolist()
# df_test['actual_labels']=actual
# # print( np.split(clf2_test_prediction,df_test.shape[0]))
# predicted=np.split(clf2_test_prediction,df_test.shape[0])
# predicted_prob=np.split(clf2_test_prediction_prob,df_test.shape[0])
# df_test['predicted_labels']=predicted
# df_test['predicted_probabilities']=predicted_prob
# print(df_test.shape)
# # predicted_arr=np.array_split(clf2_test_prediction, 5)
# # df_test['prediction_labels']=predicted_arr
# misclassified2 = clf2_test_target.values!=clf2_test_prediction
# # print(misclassified2)
# # print(type(labels_binarized_test.values))
# # # misclassified=pd.DataFrame()
# # # actual_arr=np.array_split(labels_binarized_test.values,5)
# inter=df_test[misclassified2]
# with open('new_misclassified_multiclass_153k_15k.pkl','w') as f:
#     pickle.dump(inter,f)
# # print(inter.shape)
# print(df_test.iloc[:20][:])
# # inter['predicted_labels']=
# # misclassified['YTID']=inter['YTID']
# # misclassified['labels']=inter['labels']
# # misclassified['Decision_kr']=inter['Decision_kr']
# # misclassified.to_csv('classified_kr.csv')
# # with open('new_misclassified_kr_153k_wild_64k.pkl','w') as f:
# #    pickle.dump(inter,f)
#
# print(df_test[misclassified2])


# print(clf2_test_prediction_prob)
# print(clf2_test_prediction_prob.shape)
#
# print(clf2_test_prediction.shape)
# print(clf2_test_target== clf2_test_prediction)
# print(misclassified2.shape)
# print(df_test[misclassified2])
# print(df_test[misclassified2].shape)
# print(X_test.shape)
# print(clf2_test_prediction.shape)
# print(labels_binarized_test[misclassified2].shape)
# print(labels_binarized_test[clf2_test_target].iloc[713][:]==labels_binarized_test[clf2_test_prediction].iloc[713][:])
# print(labels_binarized_test[clf1_test_prediction].head())
#

# print(clf2_test_prediction.shape)
# bin_cross_entr =(( clf2_test_target.iloc[:][0].astype(int).tolist() * log(clf2_test_prediction_prob)) + ((1- clf2_test_target.iloc.astype(int).tolist()) * (1-log(clf2_test_prediction_prob))) )/ clf2_test_target.shape[0]
# print(bin_cross_entr)
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

np1=np.array(clf2_test_target.values.tolist())
np1=np.concatenate(np1)
np2=clf2_test_prediction.tolist()
np2=np.concatenate(np2)
similarity = 1 - spatial.distance.cosine(np1,np2)
print 'Similarity: ', similarity


#print "Train Accuracy:", (clf2_train_target.values.argmax(axis=1), clf2_train_prediction.argmax(axis=1)).mean()
# print "Test Accuracy:", (clf2_test_target.values.argmax(axis=1), clf2_test_prediction.argmax(axis=1)).mean()
#
# clf2_conf_train_mat = pd.crosstab(clf2_train_target, clf2_train_prediction, margins=True)
# print("Training Precision and recall for Keras model")
# print('=============================================')
# print "Train Precision:", clf2_conf_train_mat[True][True] / float(clf2_conf_train_mat[True]['All'])
# print "Train Recall:", clf2_conf_train_mat[True][True] / float(clf2_conf_train_mat['All'][True])
# print "Train Accuracy:", (clf2_train_prediction == clf2_train_target).mean()
#
# print(clf2_conf_train_mat)
#
# clf2_conf_test_mat = pd.crosstab(clf2_test_target, clf2_test_prediction, margins=True)
# print("Testing Precision and recall for Keras model")
# print('=============================================')
#
#
# print "Test Precision:", clf2_conf_test_mat[True][True] / float(clf2_conf_test_mat[True]['All'])
# print "Test Recall:", clf2_conf_test_mat[True][True] / float(clf2_conf_test_mat['All'][True])
# print "Test Accuracy:", (clf2_test_prediction == clf2_test_target).mean()
#
# print(clf2_conf_test_mat)
# # #test loss
# bin_test_target = pd.DataFrame(clf2_test_target.astype(int))
# #print(bin_target.iloc[:][0].tolist())
# print(type(clf2_test_prediction_prob))
# bin_test_target_numpy=np.array(bin_test_target.iloc[:][0].tolist())
# print(type(bin_test_target_numpy))
# BCE_test=[]
# #train loss
# bin_train_target = pd.DataFrame(clf2_train_target.astype(int))
# #print(bin_target.iloc[:][0].tolist())
# print(type(clf2_train_prediction_prob))
# bin_train_target_numpy=np.array(bin_train_target.iloc[:][0].tolist())
# print(type(bin_train_target_numpy))
# BCE_train=[]
# #
#
# def binary_cross_entropy(y_true, y_pred):
#     return (y_true * log(y_pred)) + ((1 - y_true) * (1 - log(y_pred)))
#
# for i in zip(bin_test_target_numpy,clf2_test_prediction_prob):
#     y_test_true=i[0]
#     y_test_pred=i[1]
#     bce=binary_cross_entropy(y_test_true,y_test_pred)
#     BCE_test.append(bce)
#
# for i in zip(bin_train_target_numpy,clf2_train_prediction_prob):
#     y_train_true=i[0]
#     y_train_pred=i[1]
#     bce=binary_cross_entropy(y_train_true,y_train_pred)
#     BCE_train.append(bce)
#
# #print(BCE)
# print(np.mean(BCE_train))
# print(np.mean(BCE_test))

#model.save_weights('weights.model')
#print('Weights_saved')'''
