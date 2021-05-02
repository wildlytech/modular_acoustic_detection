# Code taken from quiuqiangkong/panns_inference/example.py


import librosa
from panns_inference import AudioTagging, labels
import numpy as np
import torch
import argparse


def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.
    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
                                  clipwise_output[sorted_indexes[k]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PANNs for predictions on audioset data")
    parser.add_argument("-a", "--audio", help="Path to audio file", required=True)
    args = parser.parse_args()
    audio_path = args.audio
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]
    print('------ Audio tagging ------')
    at = AudioTagging(checkpoint_path=None, device=device)
    (clipwise_output, embedding) = at.inference(audio)

    print_audio_tagging_result(clipwise_output[0])
