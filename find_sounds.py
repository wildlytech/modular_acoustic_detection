from subprocess import call, check_call
import argparse
import os

from pydub import AudioSegment

import numpy as np
import pandas as pd
import peakutils

def find_sound_peaks(url):
    filename = 'sounds/tmp_clip'
    filename_w_extension = filename +'.wav'

    if not os.path.exists('sounds'):
        os.makedirs('sounds')

    check_call(['youtube-dl', url, '--audio-format', 'wav', '-x', '-o', filename +'.%(ext)s'])

    window=10
    stride = 5
    range_around_peak = 5.

    aud_seg = AudioSegment.from_wav(filename_w_extension)
    aud_seg_length_ms = len(aud_seg)

    frame_rate = aud_seg.frame_rate
    num_channels = aud_seg.channels

    down_sample_rate = frame_rate/10

    # Separate the audio channels (if we have stereo)
    aud_seg = np.fromstring(aud_seg._data, np.int16).reshape((-1, num_channels)).astype(np.int32)
    aud_seg_energy = aud_seg.mean(axis=1)**2
    # aud_seg_energy = scipy.convolve(aud_seg_energy, np.ones(frame_rate/4)*4./frame_rate)

    # Aggregate all the local peaks from all the rolling-windows across the audio
    peaks = []
    for index in range(0,len(aud_seg_energy),stride*frame_rate):
        if index+window*frame_rate > len(aud_seg_energy):
            sub_seg = aud_seg_energy[index::down_sample_rate]
        else:
            sub_seg = aud_seg_energy[index:index+window*frame_rate:down_sample_rate]
        
        if len(sub_seg) == 0:
            continue
       
        # Find the peaks within the window, add offset when done
        local_peaks = peakutils.peak.indexes(sub_seg,
                                             min_dist=frame_rate*range_around_peak/2/down_sample_rate,
                                             thres=0.2)
        local_peaks = local_peaks*down_sample_rate + index
        
        # Make sure there are no overlaps between peaks from different windows
        if len(peaks) > 0:
            if len(local_peaks) > 0:
                # print local_peaks[0], peaks[-1]
                if local_peaks[0] <= peaks[-1]:
                    local_peaks = local_peaks[1:]
                elif local_peaks[0] - peaks[-1] < frame_rate*range_around_peak/2:
                    if aud_seg_energy[peaks[-1]] > aud_seg_energy[local_peaks[0]]:
                        local_peaks = local_peaks[1:]
                    else:
                        peaks = peaks[:-1]

        peaks += local_peaks.tolist()

    # Remove any duplicates
    peaks = list(set(peaks))
    peaks.sort()

    # Convert sample indexes to seconds
    peaks = np.array(peaks) / float(frame_rate)

    vfunc = np.vectorize(lambda x: np.min([aud_seg_length_ms/1000., x]))

    # compute low/high endpoints for range around peak
    start_ = (peaks - range_around_peak/2)
    end_ = vfunc(peaks + range_around_peak/2)

    # round interval endpoints to nearest half-second
    start_ = np.round(start_ / 0.5) * 0.5
    end_ = np.round(end_ / 0.5) * 0.5

    df = pd.DataFrame({'url':url,
                       'start_time':start_,
                       'end_time': end_ })
    df = df.loc[:,['url', 'start_time', 'end_time']]

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a youtube url and find time-intervals where there may be relevant sounds.')
    parser.add_argument('URL', nargs=1, help='url of youtube video')
    parser.add_argument('-o', nargs=1, dest='output_file', help='output file name (without extension)')
    args = parser.parse_args()
    url = args.URL[0]

    # url = 'https://www.youtube.com/watch?v=m5hi6bbDBm0'

    df = find_sound_peaks(url)

    if args.output_file:
        out_file = args.output_file[0] + ".csv"
        df.to_csv(out_file, index=False)
        print "\nResults written to " + out_file
    else:
        print df

