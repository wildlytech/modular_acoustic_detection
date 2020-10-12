"""
Generates the multiple target frequency components
for a single .wav file
"""
import sys
import math
import os
import argparse
import pickle
import glob
import numpy as np
import pandas as pd
import scipy.signal
from matplotlib import pyplot as plt
import scipy.io.wavfile
import resampy


#################################################################################
            # Description
#################################################################################
DESCRIPTION = 'Generates the target Goertzel filter components of audio files'



#################################################################################
            # parse the input arguments given from command line
#################################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-audio_files_path', '--audio_files_path', action='store',
                    help='input the path for audio files: .wav format files')
PARSER.add_argument('-path_to_freq_comp', '--path_to_freq_comp', action='store',
                    help='Input the path to write goertzel filter components : .pkl format files ')
RESULT = PARSER.parse_args()



#################################################################################
                # set the input arguments
#################################################################################
AUDIO_FILES_PATH = RESULT.audio_files_path
PATH_TO_GOERTZEL_COMPONENTS = RESULT.path_to_freq_comp



#################################################################################
    #set the target frequencies are interested to filter out from  audio file
#################################################################################
TARGET_FREQUENCIES = [800, 1600, 2000, 2300]
ACCEPTABLE_SAMPLINGRATE = 48000
DOWNSAMPLING_FREQUENCY = 8000


#################################################################################
            # Goertzel Algorithm implementation
#################################################################################
def Goertzel_filter(sample, sample_rate, freq, number_samples):
    """
    Implementing the goertzel filter
    """
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



#################################################################################
            # Getting the candle stick type of data
#################################################################################
def get_candlesticks_data(magnitude_data, sample_rate, time_stamp, length_of_audio_ms):

    # calculate the number of samples for evry given time stamp value
    nsamples_asper_timestamp = int((time_stamp * len(magnitude_data))/(length_of_audio_ms))
    split_data = np.split(magnitude_data, magnitude_data.shape[0]/nsamples_asper_timestamp)
    row = []
    for i in split_data:
        arb = []
        open_sample = i[0]
        close = i[-1]
        high = max(i)
        low = min(i)
        arb = [open_sample, close, high, low]
        row.append(arb)

    # Create  a dataframe with evry row as time stamp values for a single audio file
    arb_df = pd.DataFrame([[row]], columns=['Time_stamp'])

    #This is optional. If you want all the values into CSV file .
    # arb_df.to_csv('candlesticks_data_new.csv')
    return row, arb_df




#################################################################################
        # Batch processing for generating goertzel components
#################################################################################
def generate_frequency_components():
    """
    Gnerates the multiple target frequency components for
    a single audio file
    """
    # pickle_data = balancing_dataset.balanced_data(flag_for_audiomoth=0)
    # req_audio_files = pickle_data['wav_file'].tolist()
    # req_labels = pickle_data['labels_name'].tolist()
    number_audio = 0

    wavfiles_path = glob.glob(RESULT.audio_files_path + "*.wav") + glob.glob(RESULT.audio_files_path + "*.WAV")

    #############################################################################
                # Iterate through all the wav files.
                # Generate goertzel filter components
    #############################################################################
    for audio_path in wavfiles_path:
        try:
            read_file = scipy.io.wavfile.read(audio_path)


            #############################################################################
                        # Check for the sampling rate of the audio file
            #############################################################################
            if read_file[1].shape[0] == ACCEPTABLE_SAMPLINGRATE * 10:
                try:

                    #############################################################################
                                # If its stereo taking only the first of the array
                    #############################################################################
                    if read_file[1].shape[1] == 2:
                        wave_file = np.array([i[0] for i in read_file[1]])


                    #############################################################################
                                # If mono take as it is
                    #############################################################################
                except:
                    wave_file = np.array(read_file[1])

                #############################################################################
                            #Print out the details of the audio file
                #############################################################################
                number_audio = number_audio+1
                print('Audio FileName :', audio_path.split("/")[-1])
                print('Number of audio :', number_audio)

                mag = []

                #############################################################################
                         # check if the audio file is already filtered else do it
                #############################################################################
                if os.path.exists(PATH_TO_GOERTZEL_COMPONENTS + audio_path.split("/")[-1][:-4]+'.pkl'):
                    pass
                else:
                    # Resample the audio sampling rate to 8000Hz
                    wave_file = resampy.resample(wave_file, read_file[0], DOWNSAMPLING_FREQUENCY)
                    for i in TARGET_FREQUENCIES:
                        mag_arb, _, _, _ = Goertzel_filter(wave_file, DOWNSAMPLING_FREQUENCY, i, wave_file.shape[0])
                        mag.append(mag_arb)
                    with open(PATH_TO_GOERTZEL_COMPONENTS + audio_path.split("/")[-1][:-4]+'.pkl', 'wb') as file_obj:
                        pickle.dump(np.array(mag, dtype=np.float32), file_obj)
            else:
                print(' Wave file is not at good sampling rate ie ' + str(ACCEPTABLE_SAMPLINGRATE) + "Hz")

        except OSError:
            print('Wave file not found in directory '+ audio_path.split("/")[-1])



#################################################################################
                # Main function
#################################################################################
if __name__ == "__main__":
    generate_frequency_components()
