import math
import numpy as np
import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import ast
import pylab
import pydub
import glob
import os
import pickle
import scipy.signal
from pydub import AudioSegment
from matplotlib import pyplot as plt
import scipy.io.wavfile
import balance_data_priority
import arbitary
import resampy
import audiomoth


# give the path for audio files , and also path where to write the goertzel frequency components
audio_files_path = '/media/wildly/1TB-HDD/Wild_animals/'
path_to_goertzel_components = '/media/wildly/1TB-HDD/goertzel_data_8k_resampled_800_1600_2000_2300/'

#set the target target frequencies that we are interested to filter out from  audio file
target_frequencies = [800,1600,2000,2300]


def Goertzel_filter(sample,sample_rate,freq, number_samples):
    # Initialize and precomputing all the constants
    result_mag = []
    result_real = []
    result_imag =[]
    result_mag_sqre = []
    n_range = range(len(sample))
    target_freq = freq
    N = number_samples

    # computing the constants
    k = int(( N * target_freq)/sample_rate)
    w =  ((2 * math.pi * k)/N)
    cosine =  math.cos(w)
    sine = math.sin(w)
    coeff = 2 * cosine

    # Doing the calculation on the whole sample
    q1, q2 = 0.0, 0.0
    for n in range(N):
        q0  = sample[n] + coeff * q1 - q2
        q2, q1 = q1, q0

        real = ( q1 - q2 * cosine)
        imag = (q2 * sine)
        magnitude = np.square(real) + np.square(imag)
        mag_square = np.square(q1) + np.square(q2) - q1*q2*coeff
        result_real.append(real)
        result_imag.append(imag)
        result_mag.append(np.sqrt(magnitude))
        result_mag_sqre.append(mag_square)

    # return all the magnitude values
    return  result_mag, result_real, result_imag, result_mag_sqre


def get_candlesticks_data(magnitude_data, sample_rate, time_stamp,length_of_audio_ms):

    # calculate the number of samples for evry given time stamp value
    nsamples_asper_timestamp = int((time_stamp * len(magnitude_data))/(length_of_audio_ms))

    # split magnitude_data into samples for every time_stamp
    split_data = np.split(magnitude_data, magnitude_data.shape[0]/nsamples_asper_timestamp)

    n=0
    row= []
    for i in split_data :
        arb =[]
        # Start value amoung the sample
        open = i[0]
        # Last value amoung the sample
        close = i[-1]
        # Max value amoung the samples
        high = max(i)
        # Lowest value amoung the samples
        low = min(i)
        # List all the values into a single list
        arb=[open,close,high,low]
        #append the list to a main list
        row.append(arb)

    # Create  a dataframe with evry row as time stamp values for a single audio file
    arb_df = pd.DataFrame([[row]],columns=['Time_stamp'])
    #This is optional. If you want all the values into CSV file .
    # arb_df.to_csv('candlesticks_data_new.csv')
    return row, arb_df


def generate_frequency_components():

    #calling the balanced data function
    pickle_data = balance_data_priority.balancing_our_data()
    # pickle_data = audiomoth.audio_dataframe()
    print pickle_data.shape

    # pickle_data = arbitary.req_sounds()
    req_audio_files = pickle_data['wav_file'].tolist()[:]
    req_labels = pickle_data['labels_name'].tolist()[:]
    # print 'Number of audio fies: ', len(req_audio_files)

    #computation for all the wave files
    audio_files =[]
    labels=[]
    total_mag=[]
    n=0

    #Iterate through all the wav files to generate the goertzel frequency components
    for  audio,label in zip(req_audio_files,req_labels):

        #Read in the wave file
        try:
            x = scipy.io.wavfile.read(audio_files_path + audio)

            #Check for the sampling rate of the audio file
            if x[1].shape[0] ==480000 :

                #check if the audio is mono or stereo
                try:
                    # If its stereo taking only the first of the array
                    if x[1].shape[1]==2:
                        wave_file = np.array([i[0] for i in x[1]])
                        print 'wave file is stereo type'
                        print wave_file.shape

                # If the audio is mono
                except :
                    wave_file = np.array(x[1])
                    print 'Wave file is single channel'

                #Print out the details of the audio file which is undergoing goertzel filter
                print 'Audio Name :',audio
                print 'Label of Audio :',label
                print 'Number of audio :', n

                #update
                n=n+1
                mag=[]

                #check if the audio file is already filtered
                if os.path.exists( path_to_goertzel_components + audio[:11]+'.pkl'):
                    print 'Done'

                else:
                    # Resample the audio sampling rate to 8000Hz
                    print 'Executing the loop'
                    wave_file = resampy.resample(wave_file,x[0],8000)
                    print 'resamples audio: ',wave_file.shape

                    # applying Goertzel on resampled signals with required target frequecnies
                    for i  in target_frequencies :
                        mag_arb,real,imag, magnitude_square = Goertzel_filter(wave_file, 8000, i, wave_file.shape[0])
                        # mag = np.array( mag)
                        mag.append(mag_arb)
                        incr=incr+1
                    with open( path_to_goertzel_components + audio[:11]+'.pkl','w') as f:
                        pickle.dump(np.array(mag,dtype=np.float32),f)

            else:
                print ' Wave file is not at good sampling rate'
        except:
            print 'Wave file not found in directory'

    #Plot the graph for last example only
    t = np.linspace(0, 10, wave_file.shape[0])
    plt.subplot(2, 1, 1)
    plt.title('(1) wav file of 8Khz sampling rate')
    plt.xlabel('Time (seconds)')
    plt.plot(t, wave_file)

    plt.subplot(2,1,2)
    plt.title('Goertzel Filter for 2300Hz component')
    plt.xlabel('Time (seconds)')
    plt.plot(np.linspace(0, 10, wav_file.shape[0]),mag_arb)
    plt.show()

# Main function
if __name__ == "__main__":

    generate_frequency_components()
