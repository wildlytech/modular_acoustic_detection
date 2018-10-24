

#This code is copied from https://github.com/tensorflow/models/blob/master/research/audioset/vggish_inference_demo.py

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

from __future__ import print_function
import sys
import pickle
import glob
import os
import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf
sys.path.insert(0, 'externals/tensorflow_models/research/audioset/')
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim




flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'externals/tensorflow_models/research/audioset/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'externals/tensorflow_models/research/audioset/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

flags.DEFINE_string(
    'path_to_write_embeddings', None,
    'path where the embeddings should be written')

FLAGS = flags.FLAGS



def main(_):




  #Specify the path for the downloaded or recorded audio files and also path for writing the embeddings or pickle files


  # create a path if there is no path
  if not os.path.exists(FLAGS.path_to_write_embeddings):
    os.makedirs(FLAGS.path_to_write_embeddings)


  #glob all the wave files and embeddings if any .
  wav_files_path = glob.glob(FLAGS.wav_file + '*.wav')
  pickle_files = glob.glob(FLAGS.path_to_write_embeddings + '*.pkl')
  wav_file_list = []
  for wav_file_path in wav_files_path:
    wav_file_list.append(wav_file_path.split('/')[-1])


  for wav_file, wav_file_name in zip(wav_files_path, wav_file_list):
    pkl = str(wav_file_name)[:11]+'.pkl'
    print (wav_file)
    #No need to generate the embeddings that are already generated.
    if pkl in pickle_files:
        print ('Embeddings are already Generated for', pkl)


        # In this simple example, we run the examples from a single audio file through
        # the model. If none is provided, we generate a synthetic input.
    else:

      if FLAGS.wav_file:
        wav_file = wav_file
      else:
        # Write a WAV of a sine wav into an in-memory file object.
        num_secs = 5
        freq = 1000
        sampling_rate = 44100
        time_space = np.linspace(0, num_secs, int(num_secs * sampling_rate))
        theta = np.sin(2 * np.pi * freq * time_space)
        # Convert to signed 16-bit samples.
        samples = np.clip(theta * 32768, -32768, 32767).astype(np.int16)
        wav_file = six.BytesIO() 
        wavfile.write(wav_file, sampling_rate, samples)
        wav_file.seek(0)
      examples_batch = vggish_input.wavfile_to_examples(wav_file)
      print(examples_batch)

      # Prepare a postprocessor to munge the model embeddings.
      pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)


      with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        print(embedding_batch)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        print(postprocessed_batch)


        #Specify the same path that is mentioned above for writing the embeddings or pickle files
        with open(FLAGS.path_to_write_embeddings + pkl, 'w') as file_obj:
            pickle.dump(postprocessed_batch, file_obj)


if __name__ == '__main__':
  tf.app.run()
