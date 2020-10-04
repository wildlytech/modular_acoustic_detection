"""
1. To extract/ filter out the target frequency from the audio file (.wav). We do following processes
    a. Downsampling audio of any sampling rate to 8KHz. For doing it we can use any of the libraries
       in python.But in C we don't have relevant libraries for resampling / downsampling
       (on target embedded platform)
    b. So we will be implementing resampling without any libraries similar to what resampy library is doing
2. Downsampling can also be done in another way
    a. Applying Low Pass filter on the original audio (.wav) to keep cut off frequency aligning with
       the Nyquist frequency of the downsampling frequency
    b. Removing the Nth sample out the target frequency sample
4. Generating the goertzel components on the downsampled audio
"""

# To calculate Pi value
import math
# To log the time elapsed
import time
# To read the wavfile
from scipy.io import wavfile
# To initialize the lists
import numpy as np
# To resample. We will also be implementing without 
import resampy



class ReadAudioFile(object):
    """
    reads the audio file and return the samples and sampling rate
    """
    def __init__(self, path):
        self.path = path

    def read_wav_file(self):
        """
        returns sampling rate and samples
        """
        sample_rate, samples = wavfile.read(self.path)
        return sample_rate, samples


class DownsamplingImplementation(object):
    """
    implementing the resampy library with as less library as possible
    code slightly modified from :
    https://github.com/bmcfee/resampy/tree/master/resampy
    ISC License
    Copyright (c) 2016, Brian McFee
    """
    def __init__(self, samples, original_sampling_frequency):
        self.samples = samples
        self.original_sampling_frequency = original_sampling_frequency

    def load_filter(self):
        """
        loading the precomputed filter
        """

        data = np.load("kaiser_best.npz")
        return data['half_window'], data['precision'], data['rolloff']

    def resample_f(self, x, y, sample_ratio, interp_win, interp_delta, num_table):
        """
        applying interpolation
        """

        scale = min(1.0, sample_ratio)
        time_increment = 1./sample_ratio
        index_step = int(scale * num_table)
        time_register = 0.0

        n = 0
        frac = 0.0
        index_frac = 0.0
        offset = 0
        eta = 0.0
        weight = 0.0

        nwin = interp_win.shape[0]
        n_orig = x.shape[0]
        n_out = y.shape[0]
        n_channels = y.shape[1]
        print("number of n_channels: ", n_channels)

        for t in range(n_out):
            # Grab the top bits as an index to the input buffer
            n = int(time_register)

            # Grab the fractional component of the time index
            frac = scale * (time_register - n)

            # Offset into the filter
            index_frac = frac * num_table
            offset = int(index_frac)

            # Interpolation factor
            eta = index_frac - offset

            # Compute the left wing of the filter response
            i_max = min(n + 1, (nwin - offset) // index_step)
            for i in range(i_max):

                weight = (interp_win[offset + i * index_step] + eta * interp_delta[offset + i * index_step])
                for j in range(n_channels):
                    y[t, j] += weight * x[n - i, j]

            # Invert P
            frac = scale - frac

            # Offset into the filter
            index_frac = frac * num_table
            offset = int(index_frac)

            # Interpolation factor
            eta = index_frac - offset

            # Compute the right wing of the filter response
            k_max = min(n_orig - n - 1, (nwin - offset)//index_step)
            for k in range(k_max):
                weight = (interp_win[offset + k * index_step] + eta * interp_delta[offset + k * index_step])
                for j in range(n_channels):
                    y[t, j] += weight * x[n + k + 1, j]

            # Increment the time register
            time_register += time_increment



    def implement_resample(self, sr_new, axis=-1):

        if self.original_sampling_frequency <= 0:
            raise ValueError('Invalid sample rate: sr_orig={}'.format(self.original_sampling_frequency))

        if sr_new <= 0:
            raise ValueError('Invalid sample rate: sr_new={}'.format(sr_new))

        sample_ratio = float(sr_new) / self.original_sampling_frequency

        # Set up the output shape
        shape = list(self.samples.shape)
        shape[axis] = int(shape[axis] * sample_ratio)

        if shape[axis] < 1:
            raise ValueError('Input signal length={} is too small to '
                             'resample from {}->{}'.format(self.samples.shape[axis], self.original_sampling_frequency, sr_new))

        y = np.zeros(shape, dtype=self.samples.dtype)

        interp_win, precision, _ = self.load_filter()

        if sample_ratio < 1:
            interp_win *= sample_ratio
        

        # create a numpy array of zeros similar to shape of interpolation window i.e interp_win
        interp_delta = np.zeros_like(interp_win)

        # subtract the successive element with the preceding element in the array i.e np.diff
        # a[n] = a[n+1] - a[n]
        interp_delta[:-1] = np.diff(interp_win)

        # Construct 2d views of the data with the resampling axis on the first dimension
        x_2d = self.samples.reshape((self.samples.shape[axis], -1))
        y_2d = y.reshape((y.shape[axis], -1))
        print("Process Initiated")
        self.resample_f(x_2d,y_2d, sample_ratio, interp_win, interp_delta, precision)

        return y


class DownsampleUsingLibrary(object):
    """
    here we downsample the audio using the
    """
    def __init__(self, samples, original_sampling_frequency):
        self.samples = samples
        self.original_sampling_frequency = original_sampling_frequency

    def resample_using_resampy(self, target_sampling_frequency):
        """
        resamples the audio to required frequency using resampy library
        """
        return resampy.resample(self.samples, self.original_sampling_frequency, target_sampling_frequency)



class GoertzelComponents(object):
    """
    executes goertzel filtering on audio
    """
    def __init__(self, samples, sample_rate, target_frequency, number_samples ):
        self.samples = samples
        self.sample_rate = sample_rate
        self.target_frequency = target_frequency
        self.number_samples = number_samples


    def goertzel_filter(self):
        """
        Implements the goertzel algorithm and
        returs the target frequency components of the audio
        """
        # Initialize and precomputing all the constants
        result_mag = np.zeros((self.sample_rate)).tolist()
        total_samples = self.number_samples

        # computing the constants
        k_constant = int((total_samples * self.target_frequency)/self.sample_rate)
        w_constant = ((2 * math.pi * k_constant)/total_samples)
        cosine = math.cos(w_constant)
        sine = math.sin(w_constant)
        coeff = 2 * cosine

        # Doing the calculation on the whole sample
        q_1, q_2 = 0.0, 0.0

        index = 0
        for n_sample in range(total_samples):
            q_0 = self.samples[n_sample] + coeff * q_1 - q_2
            q_2, q_1 = q_1, q_0

            real = (q_1 - q_2 * cosine)
            imag = (q_2 * sine)
            magnitude = np.square(real) + np.square(imag) 
            result_mag[index] = np.sqrt(magnitude)
            index += 1
        return  result_mag

