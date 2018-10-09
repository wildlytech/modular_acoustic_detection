import math
import numpy as np
import pandas as pd
import numpy as np
import math
import scipy.signal
from matplotlib import pyplot as plt
import scipy.io.wavfile
import time
import math
import resampy

#set the resampling rate, target frequency
resampling_rate = 8000
target_frequency = 400
number_of_seconds = 10

#read the sample wav file
x = scipy.io.wavfile.read('speech.wav')
wave_file = np.array([i[0] for i in x[1]])
print wave_file.shape

#resampling the wave file
wave_file =resampy.resample(wave_file,44100,resampling_rate)
print wave_file.shape

def Goertzel_filter(sample,sample_rate,target_frequency, number_samples):

    # Initialize and precomputing all the constants
    result_mag = np.zeros((sample_rate*10,1))
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

    i=0
    start=time.time()
    for n in range(N):
        q0  = sample[n] + coeff * q1 - q2
        q2, q1 = q1, q0

        real = ( q1 - q2 * cosine)
        imag = (q2 * sine)
        magnitude = np.square(real) + np.square(imag)
        result_mag[i]=math.sqrt(magnitude)
    	i+=1
    end=time.time()
    print 'Time elapsed :', (end-start)
    return  result_mag

# applying Goertzel on those signals
mag = Goertzel_filter(wave_file, resampling_rate, target_frequency, resampling_rate*numer_of_seconds)
mag = map(int, mag)
SAMPLE_RATE = 44100
t = np.linspace(0, 10, resampling_rate*10)

#PLOT THE GOERTZEL FILTER COMPONENT
plt.subplot(2, 1, 1)
plt.title('(1) speech wave of 44.1KHz sampling rate')
plt.xlabel('Time (seconds)')
plt.plot(t, wave_file)

plt.subplot(2,1,2)
plt.title('Goertzel Filter for 2000Hz component')
plt.xlabel('Time (seconds)')
plt.plot(np.linspace(0, 10, resampling_rate*10),mag)
plt.show()
