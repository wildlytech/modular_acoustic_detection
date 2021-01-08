"""
Implements the goertzel filter algorithm that
returns the target frequency components of the audio
"""
import math
import time
import argparse
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
import scipy.io.wavfile
import resampy


###############################################################################
# Description and Help
###############################################################################
DESCRIPTION = 'Input the path of audio file \
              and target frequency to filter '
HELP = 'Give relevant Inputs'


###############################################################################
# Parse the arguments
###############################################################################
PARSER = argparse.ArgumentParser(description=DESCRIPTION)
PARSER.add_argument('-target_frequency_to_filter', '--target_frequency_to_filter',
                    action='store', type=int, help='Input the frequency (in Hz)')
PARSER.add_argument('-wavfile_path_to_filter', '--wavfile_path_to_filter',
                    action='store', help='Input the path (.wav file)')
RESULT = PARSER.parse_args()


###############################################################################
# set the resampling rate, target frequency
###############################################################################

RESAMPLING_RATE = 8000
TARGET_FREQUENCY = RESULT.target_frequency_to_filter
NUMBER_OF_SECONDS = 10


###############################################################################
# read the sample wav file
###############################################################################
SAMPLE_RATE, READ_FILE = scipy.io.wavfile.read(RESULT.wavfile_path_to_filter)
WAV_FILE = np.array([i[0] for i in READ_FILE])
print(WAV_FILE.shape)


###############################################################################
# resampling the wave file
###############################################################################
WAV_FILE = resampy.resample(WAV_FILE, SAMPLE_RATE, RESAMPLING_RATE)
print(WAV_FILE.shape)


###############################################################################
# Goertzel Implementation
###############################################################################
def Goertzel_filter(sample, sample_rate, target_frequency, number_samples):
    """
    Implements the goertzel algorithm and
    returs the target frequency components of the audio
    """
    # Initialize and precomputing all the constants
    result_mag = np.zeros((sample_rate*10, 1))
    total_samples = number_samples

    # computing the constants
    k_constant = int((total_samples * target_frequency)/sample_rate)
    w_constant = ((2 * math.pi * k_constant)/total_samples)
    cosine = math.cos(w_constant)
    sine = math.sin(w_constant)
    coeff = 2 * cosine

    # Doing the calculation on the whole sample
    q_1, q_2 = 0.0, 0.0

    index = 0
    start = time.time()
    for n_sample in range(total_samples):
        q_0 = sample[n_sample] + coeff * q_1 - q_2
        q_2, q_1 = q_1, q_0

        real = (q_1 - q_2 * cosine)
        imag = (q_2 * sine)
        magnitude = np.square(real) + np.square(imag)
        result_mag[index] = math.sqrt(magnitude)
        index += 1
    end = time.time()
    print('Time elapsed :', (end-start))
    return  result_mag


###############################################################################
# applying Goertzel on those signals
###############################################################################
MAGNITUDE = Goertzel_filter(WAV_FILE, RESAMPLING_RATE,
                            TARGET_FREQUENCY, RESAMPLING_RATE*NUMBER_OF_SECONDS)
MAGNITUDE = list(map(int, MAGNITUDE))
SAMPLE_RATE = SAMPLE_RATE
TIME_SPACE = np.linspace(0, NUMBER_OF_SECONDS, RESAMPLING_RATE*10)


###############################################################################
# plot the goertzel filter components
###############################################################################
plt.subplot(2, 1, 1)
plt.title('(1) speech wave of 44.1KHz sampling rate')
plt.xlabel('Time (seconds)')
plt.plot(TIME_SPACE, WAV_FILE)

plt.subplot(2, 1, 2)
plt.title('Goertzel Filter for '+ str(TARGET_FREQUENCY) + 'HZ component')
plt.xlabel('Time (seconds)')
plt.plot(np.linspace(0, NUMBER_OF_SECONDS, RESAMPLING_RATE*NUMBER_OF_SECONDS), MAGNITUDE)
plt.show()
