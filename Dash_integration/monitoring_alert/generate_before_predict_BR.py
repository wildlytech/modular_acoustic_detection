
#This code is copied from
#https://github.com/tensorflow/models/blob/master/research/audioset/vggish_inference_demo.py

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
import tensorflow as tf
from keras import backend as K
sys.path.insert(0, '../../externals/tensorflow_models/research/audioset/')
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import model_function_binary_relevance


flags = tf.app.flags

flags.DEFINE_string(
    'checkpoint', '../../externals/tensorflow_models/research/audioset/vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')


flags.DEFINE_string(
    'local_folder_path', None, help='Path to the VGGish checkpoint file.')
flags.DEFINE_string(
    'remote_ftp_path', None, help='Path to the VGGish checkpoint file.')
flags.DEFINE_string(
    'csv_filename', None, help='Path to the VGGish checkpoint file.')
flags.DEFINE_string(
    'ftp_username', None, help='Path to the VGGish checkpoint file.')
flags.DEFINE_string(
    'ftp_host', None, help='Path to the VGGish checkpoint file.')
flags.DEFINE_string(
    'ftp_password', None, help='Path to the VGGish checkpoint file.')



flags.DEFINE_string(
    'pca_params', '../../externals/tensorflow_models/research/audioset/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS


def main(wav_file, flag_for_data, data, model_type):
    """
    #Specify the path for the downloaded or recorded audio files and
    #also path for writing the embeddings or pickle files
    """
    if flag_for_data == 0:
        if wav_file:
            pkl = wav_file[:-4]+'.pkl'
          # print (pkl)
        examples_batch = vggish_input.wavfile_to_examples(wav_file)

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
            postprocessed_batch = pproc.postprocess(embedding_batch)
            return postprocessed_batch
        # print(postprocessed_batch)
    elif flag_for_data == 1:
        predict_prob, predictions = model_function_binary_relevance.predictions_wavfile(data, model_type)
        print(predict_prob, predictions)
        K.clear_session()
        return predict_prob, predictions
