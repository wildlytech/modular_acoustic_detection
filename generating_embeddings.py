

#This code is modified from :
#https://github.com/tensorflow/models/blob/master/research/audioset/vggish/vggish_inference_demo.py

r"""A simple demonstration of running VGGish in inference mode.
This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.
A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""


import argparse
import sys
import pickle
import glob
import os
import numpy as np
from scipy.io import wavfile
import six
import tensorflow.compat.v1 as tf

VGGISH_PATH = 'externals/tensorflow_models/research/audioset/vggish/'
sys.path.insert(0, VGGISH_PATH)

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim



#######################################################################################
                    # Constants
#######################################################################################

VGGISH_MODEL_CHECKPOINT_FILE = VGGISH_PATH + 'vggish_model.ckpt'
PCA_PARAMS_FILE = VGGISH_PATH + 'vggish_pca_params.npz'

########################################################################################
                    # Functions
########################################################################################

def generate(path_to_write_embeddings, path_to_wav_files):
    """
    Generates the embeddings for the wav files at specified path
    """
    # append a slash at the end of the path to read wav files
    # if it is not there already
    if not path_to_wav_files.endswith('/'):
        path_to_wav_files += '/'

    # append a slash at the end of the path to generate embeddings
    # if it is not there already
    if not path_to_write_embeddings.endswith('/'):
        path_to_write_embeddings += '/'

    # create a path if there is no path
    if not os.path.exists(path_to_write_embeddings):
        os.makedirs(path_to_write_embeddings)

    #glob all the wave files and embeddings if any .
    wav_files_path = glob.glob(path_to_wav_files + '*.wav')
    pickle_files = glob.glob(path_to_write_embeddings + '*.pkl')
    wav_file_list = []
    for wav_file_path in wav_files_path:
        wav_file_list.append(wav_file_path.split('/')[-1])

    for wav_file, wav_file_name in zip(wav_files_path, wav_file_list):
        path_to_pickle_file = path_to_write_embeddings + str(wav_file_name)[:-4] + '.pkl'
        print (wav_file)
        #No need to generate the embeddings that are already generated.
        if path_to_pickle_file in pickle_files:
            print ('Embeddings are already generated for', path_to_pickle_file)

        # In this simple example, we run the examples from a single audio file through
        # the model. If none is provided, we generate a synthetic input.
        else:
            examples_batch = vggish_input.wavfile_to_examples(wav_file)
            # print(examples_batch)

            # Prepare a postprocessor to munge the model embeddings.
            pproc = vggish_postprocess.Postprocessor(PCA_PARAMS_FILE)

            with tf.Graph().as_default(), tf.Session() as sess:
                # Define the model in inference mode, load the checkpoint, and
                # locate input and output tensors.
                vggish_slim.define_vggish_slim(training=False)
                vggish_slim.load_vggish_slim_checkpoint(sess, VGGISH_MODEL_CHECKPOINT_FILE)
                features_tensor = sess.graph.get_tensor_by_name(
                    vggish_params.INPUT_TENSOR_NAME)
                embedding_tensor = sess.graph.get_tensor_by_name(
                    vggish_params.OUTPUT_TENSOR_NAME)

                # Run inference and postprocessing.
                [embedding_batch] = sess.run([embedding_tensor],
                                             feed_dict={features_tensor: examples_batch})
                # print(embedding_batch)
                postprocessed_batch = pproc.postprocess(embedding_batch)
                # print(postprocessed_batch)

            #write the embeddings as pickle files
            with open(path_to_pickle_file, 'wb') as file_obj:
                pickle.dump(postprocessed_batch, file_obj)

if __name__ == '__main__':
    DESCRIPTION = 'Generate Embeddings for wav files'
    PARSER = argparse.ArgumentParser(description=DESCRIPTION)
    REQUIRED_ARGUMENTS = PARSER.add_argument_group('required arguments')
    REQUIRED_ARGUMENTS.add_argument('-wav_file', '--wav_file',
                                    action='store',
                                    help='Path to directory with wav files',
                                    required=True)
    REQUIRED_ARGUMENTS.add_argument('-path_to_write_embeddings', '--path_to_write_embeddings',
                                    action='store',
                                    help='Output path to save pkl files',
                                    required=True)
    RESULT = PARSER.parse_args()


    generate(path_to_write_embeddings = RESULT.path_to_write_embeddings,
             path_to_wav_files = RESULT.wav_file)
