"""
Generates the multiple target frequency components
for a single .wav file
"""
import math
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.signal
from matplotlib import pyplot as plt
import scipy.io.wavfile
import balance_data_priority
import resampy


DESCRIPTION = 'Generates the target Goertzel filter components of audio files'

#parse the input arguments given from command line
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-audio_files_path', '--audio_files_path', action='store',
                    help='input the path for audio files: .wav format files')
PARSER.add_argument('-path_to_freq_comp', '--path_to_freq_comp', action='store',
                    help='Input the path to write goertzel filter components : .pkl format files ')
RESULT = PARSER.parse_args()

# give the path for audio files , and also path where to write the goertzel frequency components
AUDIO_FILES_PATH = RESULT.audio_files_path
PATH_TO_GOERTZEL_COMPONENTS = RESULT.path_to_freq_comp

#set the target target frequencies that we are interested to filter out from  audio file
TARGET_FREQUENCIES = [800, 1600, 2000, 2300]


def Goertzel_filter(sample, sample_rate, freq, number_samples):
    """
    Implementing the goertzel filter
    """
    # Initialize and precomputing all the constants
    result_mag = []
    result_real = []
    result_imag = []
    result_mag_sqre = []
    target_freq = freq
    total_samples = number_samples

    # computing the constants
    k_constant = int((total_samples * target_freq)/sample_rate)
    w_constant = ((2 * math.pi * k_constant)/total_samples)
    cosine = math.cos(w_constant)
    sine = math.sin(w_constant)
    coeff = 2 * cosine

    # Doing the calculation on the whole sample
    q_1, q_2 = 0.0, 0.0
    for n_sample in range(total_samples):
        q_0 = sample[n_sample] + coeff * q_1 - q_2
        q_2, q_1 = q_1, q_0

        real = (q_1 - q_2 * cosine)
        imag = (q_2 * sine)
        magnitude = np.square(real) + np.square(imag)
        mag_square = np.square(q_1) + np.square(q_2) - q_1*q_2*coeff
        result_real.append(real)
        result_imag.append(imag)
        result_mag.append(np.sqrt(magnitude))
        result_mag_sqre.append(mag_square)

    # return all the magnitude values
    return  result_mag, result_real, result_imag, result_mag_sqre


def get_candlesticks_data(magnitude_data, sample_rate, time_stamp, length_of_audio_ms):

    # calculate the number of samples for evry given time stamp value
    nsamples_asper_timestamp = int((time_stamp * len(magnitude_data))/(length_of_audio_ms))

    # split magnitude_data into samples for every time_stamp
    split_data = np.split(magnitude_data, magnitude_data.shape[0]/nsamples_asper_timestamp)
    row = []
    for i in split_data:
        arb = []
        # Start value amoung the sample
        open_sample = i[0]
        # Last value amoung the sample
        close = i[-1]
        # Max value amoung the samples
        high = max(i)
        # Lowest value amoung the samples
        low = min(i)
        # List all the values into a single list
        arb = [open_sample, close, high, low]
        #append the list to a main list
        row.append(arb)

    # Create  a dataframe with evry row as time stamp values for a single audio file
    arb_df = pd.DataFrame([[row]], columns=['Time_stamp'])

    #This is optional. If you want all the values into CSV file .
    # arb_df.to_csv('candlesticks_data_new.csv')
    return row, arb_df


def generate_frequency_components():
    """
    Gnerates the multiple target frequency components for
    a single audio file
    """
    #calling the balanced data function
    pickle_data = balance_data_priority.balancing_our_data()
    # pickle_data = audiomoth.audio_dataframe()
    print pickle_data.shape

    # pickle_data = arbitary.req_sounds()
    req_audio_files = pickle_data['wav_file'].tolist()[:]
    req_labels = pickle_data['labels_name'].tolist()[:]

    #computation for all the wave files
    number_audio = 0

    #Iterate through all the wav files to generate the goertzel frequency components
    for audio, label in zip(req_audio_files, req_labels):

        #Read in the wave file
        try:
            read_file = scipy.io.wavfile.read(AUDIO_FILES_PATH + audio)

            #Check for the sampling rate of the audio file
            if read_file[1].shape[0] == 480000:

                #check if the audio is mono or stereo
                try:
                    # If its stereo taking only the first of the array
                    if read_file[1].shape[1] == 2:
                        wave_file = np.array([i[0] for i in read_file[1]])
                        print 'wave file is stereo type'
                        print wave_file.shape

                # If the audio is mono
                except:
                    wave_file = np.array(read_file[1])
                    print 'Wave file is single channel'

                #Print out the details of the audio file which is undergoing goertzel filter
                print 'Audio Name :', audio
                print 'Label of Audio :', label
                print 'Number of audio :', number_audio

                #update
                number_audio = number_audio+1
                mag = []

                #check if the audio file is already filtered
                if os.path.exists(PATH_TO_GOERTZEL_COMPONENTS + audio[:11]+'.pkl'):
                    print 'Done'

                else:
                    # Resample the audio sampling rate to 8000Hz
                    print 'Executing the loop'
                    wave_file = resampy.resample(wave_file, x[0], 8000)
                    print 'resamples audio: ', wave_file.shape

                    # applying Goertzel on resampled signals with required target frequecnies
                    for i in TARGET_FREQUENCIES:
                        mag_arb, real, imag, magnitude_square = Goertzel_filter(wave_file, 8000, i, wave_file.shape[0])
                        mag.append(mag_arb)
                    with open(PATH_TO_GOERTZEL_COMPONENTS + audio[:11]+'.pkl', 'w') as file_obj:
                        pickle.dump(np.array(mag, dtype=np.float32), file_obj)

            else:
                print ' Wave file is not at good sampling rate'
        except:
            print 'Wave file not found in directory'

    #Plot the graph for last example only
    time_space = np.linspace(0, 10, wave_file.shape[0])
    plt.subplot(2, 1, 1)
    plt.title('(1) wav file of 8Khz sampling rate')
    plt.xlabel('Time (seconds)')
    plt.plot(time_space, wave_file)

    plt.subplot(2, 1, 2)
    plt.title('Goertzel Filter for 2300Hz component')
    plt.xlabel('Time (seconds)')
    plt.plot(np.linspace(0, 10, wave_file.shape[0]), mag_arb)
    plt.show()

# Main function
if __name__ == "__main__":
    generate_frequency_components()
