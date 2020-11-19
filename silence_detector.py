import scipy
import librosa
import numpy as np
import argparse

def read_soundfile(filepath):
    """
    Reads sound file from given path:

    Parameters:
    filepath,str: Path to file to be read

    Returns:
    sig, numpy array: Signal read from the file
    fs,float: Sampling rate at which the signal was sampled
    """
    sig, fs = librosa.load(filepath)
    return sig,fs

def create_spec(sig,fs):
    """
    Reads signal and frame rate and creates a spectrogram

    Parameters:
        sig,np.ndarray: Signal for which to create the spectrogram
        fs,float: Signal's frame rate

    Returns:
        pdb,librosa spectrogram: Spectrogram converted form power to decibel scale
    """
    f, t, sxx = scipy.signal.spectrogram(sig, fs)
    pdb = librosa.power_to_db(sxx, ref=np.max)
    return pdb,t


def silence_detector(pdb,t, thresh=-60):
    """
    Reads the spectrogram and detects silences

    Parameters:
        pdb: Spectrogram in decibel scale
        thresh: Threshold below which the sound is considered to be silent.

    Returns:
        time_silences,List: List of time co-ordinates at which silence occurs.
    """
    mask = pdb > thresh
    beg = 0

    silence_limits = []
    for i in range(mask.shape[1]):
        if sum(mask[:, i]) == 0:
            if beg == 0:
                start = i
                end = i
                beg = 1
            else:
                end = i

        else:
            if (start, end) not in silence_limits:
                silence_limits.append((start, end))
            beg = 0

    time_silence = [(t[tup[0]], t[tup[1]]) for tup in silence_limits]
    return time_silence

if __name__=="__main__":
    description = "Detects silences in a spectrogram"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-filepath","--filepath",action="store",help="Input path to the file",required=True)
    parser.add_argument("-thresh","--thresh",action="store",help="threshold below which sound is silence")

    args = parser.parse_args()
    sig,fs = read_soundfile(args.filepath)
    pdb,t = create_spec(sig,fs)
    if args.thresh:
        time_silence = silence_detector(pdb, t, args.thresh)
    else:
        time_silence = silence_detector(pdb,t)
    print(time_silence)
