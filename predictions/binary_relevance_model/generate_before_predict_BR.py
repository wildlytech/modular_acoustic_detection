
#This code is copied and modified from
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


import sys
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
from . import model_function_binary_relevance

VGGISH_PATH = 'externals/tensorflow_models/research/audioset/vggish/'
sys.path.insert(0, VGGISH_PATH)

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim


##############################################################################
    # Defining the flags before hand
##############################################################################
FLAGS = {
    'pca_params' : VGGISH_PATH + 'vggish_pca_params.npz',
    'checkpoint' : VGGISH_PATH + 'vggish_model.ckpt'
}

##############################################################################

##############################################################################
def main(wav_file, flag_for_data, data,model_type):
    """
    Specify the path for the downloaded or recorded audio files and
    also path for writing the embeddings or pickle files
    """
    if flag_for_data == 0:
        examples_batch = vggish_input.wavfile_to_examples(wav_file)


        # Prepare a postprocessor to munge the model embeddings.
        pproc = vggish_postprocess.Postprocessor(FLAGS['pca_params'])

        with tf.Graph().as_default(), tf.Session() as sess:

            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS['checkpoint'])
            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: examples_batch})
            postprocessed_batch = pproc.postprocess(embedding_batch)
            # print(postprocessed_batch)
            return postprocessed_batch
    elif flag_for_data == 1:
        predict_prob, predictions = model_function_binary_relevance.predictions_wavfile(data, model_type)
        K.clear_session()
        return predict_prob, predictions
