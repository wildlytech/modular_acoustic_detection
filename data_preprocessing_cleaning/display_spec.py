import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import argparse


def get_spectrogram(filepath):
    """
    Creates mel spectrogram using Librosa

    Parameters:
        filepath: str, Path to input sound file
    Returns:
        S: numpy array: Spectrogram as calculated by Librosa
    """
    sig, fs = librosa.load(filepath)
    S = librosa.feature.melspectrogram(y=sig, sr=fs)
    return S


def plot_spec(S, filepath=None):
    """
    Create plot and save the same in the given output filepath

    Parameters:
        S: numpy array, Spectrogram calculated by Librosa
        filepath: str,Output filepath where to save the plot


    """
    ax = plt.subplot(111)
    fig = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis="time", y_axis="log", ax=ax)
    plt.colorbar(fig, ax=ax)
    if filepath != None:
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


if __name__ == "__main__":
    DESCRIPTION = "SPECTROGRAM DISPLAY MODULE"
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    REQUIRED_ARGUMENTS = PARSER.add_argument_group("required arguments")
    OPTIONAL_ARGUMENTS = PARSER.add_argument_group("optional arguments")
    REQUIRED_ARGUMENTS.add_argument("-input_file_path", action="store", help="Input filepath to sound file", required=True)
    OPTIONAL_ARGUMENTS.add_argument("-output_file_path", action="store", help="Output file path of location where plot is to be saved")
    RESULT = PARSER.parse_args()

    S = get_spectrogram(RESULT.input_file_path)
    if RESULT.output_file_path:
        plot_spec(S, RESULT.output_file_path)
    else:
        plot_spec(S)
