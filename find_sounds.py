from subprocess import call, check_call
import argparse
import os

from pydub import AudioSegment

import numpy as np
import pandas as pd
import peakutils

import librosa

def find_sound_peaks(filepath, url):

    window=10
    stride = 5
    range_around_peak = 5.
    num_frequency_bands = 8

    aud_seg = AudioSegment.from_wav(filepath)
    aud_seg_length_ms = len(aud_seg)

    frame_rate = aud_seg.frame_rate
    num_channels = aud_seg.channels

    down_sample_rate = frame_rate/10

    # Separate the audio channels (if we have stereo)
    aud_seg = np.fromstring(aud_seg._data, np.int16).reshape((-1, num_channels)).astype(np.int32)

    # Take the short-time fourier transform of the mean of the audio channels.
    # We expect the audio channels to be mostly the same.
    stft_ = librosa.stft(aud_seg.mean(axis=1))

    # Decompose the stft into different bands and take the inverse stft to get the energy of the
    # component of the original signal pertaining to each respective band
    stft_bands_ = []
    power_bands_ = []
    for i in range(num_frequency_bands):
        stft_band_ = stft_.copy()
        if i > 0:
            stft_band_[:3*(2**(i-1)),:] = 0
        stft_band_[3*(2**i):,:] = 0
        
        stft_bands_ += [stft_band_]
        power_bands_ += [librosa.istft(stft_band_)**2]

    # Normalize each component to [0,1] scale and essentially perform an additive-OR of all of them
    # The composite signal that is generated will be used for peak detection
    aud_seg_energy = []
    for index in range(len(power_bands_)):
        aud_seg_energy += [power_bands_[index] / power_bands_[index].max()]
    aud_seg_energy = np.array(aud_seg_energy).max(axis=0)

    # Aggregate all the local peaks from all the rolling-windows across the audio
    peaks = []
    for index in range(0,len(aud_seg_energy),stride*frame_rate):
        if index+window*frame_rate > len(aud_seg_energy):
            sub_seg = aud_seg_energy[index::down_sample_rate]
        else:
            sub_seg = aud_seg_energy[index:index+window*frame_rate:down_sample_rate]
        
        if len(sub_seg) == 0:
            continue

        # the function in peakutils doesn't work if array is constant
        if sum(sub_seg - sub_seg.mean()) == 0:
            continue
       
        # Find the peaks within the window, add offset when done
        local_peaks = peakutils.peak.indexes(sub_seg,
                                             min_dist=frame_rate*range_around_peak/2/down_sample_rate,
                                             thres=0.2)

        local_peaks = local_peaks*down_sample_rate + index

        if index > 0:
            # Make sure there are no overlaps between peaks from different windows
            while len(peaks) > 0 and len(local_peaks) > 0:
                # print local_peaks[0], peaks[-1]
                if local_peaks[0] <= peaks[-1]:
                    local_peaks = local_peaks[1:]
                elif local_peaks[0] - peaks[-1] < frame_rate*range_around_peak/2:
                    if aud_seg_energy[peaks[-1]] > aud_seg_energy[local_peaks[0]]:
                        local_peaks = local_peaks[1:]
                    else:
                        peaks = peaks[:-1]
                else:
                    break

        peaks += local_peaks.tolist()

    # Remove any duplicates
    peaks = list(set(peaks))
    peaks.sort()

    # Convert sample indexes to seconds
    peaks = np.array(peaks) / float(frame_rate)

    df = pd.DataFrame({'url':url,
                       'time':peaks
                       })
    df = df.loc[:,['url', 'time']]

    return df

def download_youtube_url(url):
    filename = 'sounds/tmp_clip'
    filename_w_extension = filename +'.wav'

    if not os.path.exists('sounds'):
        os.makedirs('sounds')

    check_call(['youtube-dl', url, '--audio-format', 'wav', '-x', '-o', filename +'.%(ext)s'])

    return filename_w_extension

def extract_sound_clips(filepath, df, directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

    aud_seg = AudioSegment.from_wav(filepath)

    for time in df.time:
        start_time = np.max([0,time-2.5])
        end_time = np.min([time+2.5, len(aud_seg)/1000])
        aud_seg[int(start_time*1000):int(end_time*1000)].export(directory + '/sound-' + str(start_time) + '-' + str(end_time)+".wav", format="wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a youtube url and find time-intervals where there may be relevant sounds.')
    parser.add_argument('URL', nargs=1, help='url of youtube video')
    parser.add_argument('-i', nargs=1, dest='input_file', help='input file path (to use if you don\'t want to download url and just use the file from disk)')
    parser.add_argument('-o', nargs=1, dest='output_file', help='output file name (without file extension)')
    parser.add_argument('-e', nargs=1, dest='extract_dir', help='extract relevant sound clips to subdirectory')
    args = parser.parse_args()
    url = args.URL[0]

    # url = 'https://www.youtube.com/watch?v=m5hi6bbDBm0'

    if args.input_file:
        filename_w_extension = args.input_file[0]
    else:
        filename_w_extension = download_youtube_url(url)
    df = find_sound_peaks(filename_w_extension, url)

    if args.output_file:
        out_file = args.output_file[0] + ".csv"
        df.to_csv(out_file, index=False)
        print "\nResults written to " + out_file
    else:
        print df

    if args.extract_dir:
        extract_sound_clips(filename_w_extension, df, args.extract_dir[0])
        print "Sound clips extracted!"

