"""
Predictions are done using pre-trained weights of
goertzel model and without using any of the higher level python libraries.
"""
# To give out stderr
import sys
# To compute exponential value ( sigmoid function )
import math
# To generate random values as input test data
import numpy as np
# To read the csv file
import pandas as pd
# To read the weights file
import pickle
# To print colored text in terminal. Useful while debugging
from colorama import Fore, Style
# To log the time lapse for computation
import time
# To downsample and apply goertzel filter
import downsampling_and_goertzel_filter


# Read all the global variables i.e weights and Inputs. 
with open("motor_weights_goertzel_remodel_400_500_2000_2300_maxpool_frst_kern_8_432305555.pkl", "rb") as f:
    WEIGHTS_VALUE = pickle.load(f)
ARCHITECTURE_CSV = pd.read_csv("test_architecture.csv")
TEN_SEC_INPUT_DATA = np.random.uniform(low=0.00000002, high=0.00001, size=(10, 8000, 4))


class ConvolutionalNeuralNetwork(object):

    """
    implementing neural network
    without any libraries
    """

    def __init__(self, units, input_data, weights, activation, bias):
        self.units = units
        self.input_data = input_data
        self.weights = weights
        self.activation = activation
        self.bias = bias


    def padding(self, filter_size, stride_length, type_padding):
        """
        apply padding to the input data
        """
        if type_padding == "same" and stride_length > 1:
            self.input_data = self.input_data + [[0 for cols in range(len(self.input_data[0]))]for rows in range(stride_length)]
            return np.array(self.input_data, dtype='float64').tolist()
        else:
            self.input_data = self.input_data + [[0 for col in range(len(self.input_data[0]))] for row in range(filter_size-1)]
            return np.array(self.input_data, dtype='float64').tolist()


    def matrix_multiplication(self, data, nested_index):
        """
        implement matrix multiplication
        """
        result = [[0] * len(self.weights[nested_index][0])]
        # result = np.zeros((1, len(self.weights[nested_index][0]))).tolist()
        # Iterate thorugh rows of X
        for i in range(len(data)):
           # Iterate through columns of Y
            for j in range(len(self.weights[nested_index][0])):
               # Iterate through rows of Y
                for k in range(len(self.weights[nested_index])):
                    result[i][j] += data[i][k] * self.weights[nested_index][k][j]

        return  np.array(result, dtype='float64').tolist()

    def add_bias_terms(self, data):
        """
        adds bias terms to the data
        """
        try:
            add_bias = [sum(zip_values) for zip_values in zip(data, self.bias)]
            print "not except"
        except TypeError:
            add_bias = [sum(zip_values) for zip_values in zip(data[0], self.bias)]
        return np.array(add_bias, dtype='float64').tolist()

    def relu(self, data):
        """
        implements matrix multiplication
        and returns relu of that
        """
        try:
            apply_relu = [0 if arb < 0 else arb for arb in data]
        except (TypeError, ValueError) as error:
            apply_relu = [0 if arb < 0 else arb for arb in data[0]]
        return np.array(apply_relu, dtype='float64').tolist()



    def get_index_for_window_movement(self, stride_length):
        """
        returns the index values as per the strides
        """
        return range(0, len(self.input_data), stride_length)


    def start_window_moment(self, stride_length, filter_size, type_padding):
        """
        implements covolutional multiplication
        """
        result_final = np.zeros((len(self.input_data)/stride_length, self.units)).tolist()
        initial_input_data_length = len(self.input_data)/stride_length
        self.input_data = self.padding(filter_size, stride_length, type_padding)
        index_range = self.get_index_for_window_movement(stride_length)


        for inter_index, index in enumerate(index_range[:initial_input_data_length]):
            for nested_index, data_index in enumerate(range(index, index+filter_size)):
                if  nested_index == 0:
                    result_after_mat_mull = []
                    result_after_mat_mull = self.matrix_multiplication([self.input_data[data_index]], nested_index)
                else:
                    result_after_mat_mull = [sum(example) for example in zip(result_after_mat_mull[0],
                                                                             self.matrix_multiplication([self.input_data[data_index]], nested_index)[0])]
                    result_after_mat_mull = [result_after_mat_mull]
            result_after_mat_mull = self.add_bias_terms(result_after_mat_mull)
            result_final[inter_index] = np.array(self.relu(result_after_mat_mull), dtype='float64').tolist()

        return np.array(result_final, dtype='float64').tolist()



class FullyConnectedLayer(object):

    """
    implements dense layer computations
    """
    def __init__(self, units, input_data, weights, activation, bias):
        self.units = units
        self.input_data = input_data
        # self.input_shape = input_data.shape
        self.weights = weights
        self.activation = activation
        self.bias = bias

    def matrix_multiplication(self, data):
        """
        implement matrix multiplication
        """

        result = np.zeros((len(data), self.units)).tolist()
        # Iterate thorugh rows of X
        for i in range(len(data)):
           # Iterate through columns of Y
            for j in range(len(self.weights[0])):
               # Iterate through rows of Y
                for k in range(len(self.weights)):
                    # print len(self.weights[0])
                    result[i][j] += data[i][k] * self.weights[k][j]

        return  np.array(result, dtype='float64').tolist()

    def add_bias_terms(self, data):
        """
        adds bias terms to the data
        """
        try:
            add_bias = [sum(zip_values) for zip_values in zip(data, self.bias)]
        except TypeError:
            add_bias = [sum(zip_values) for zip_values in zip(data[0], self.bias)]
        return np.array(add_bias, dtype='float64').tolist()


    def relu(self, data):
        """
        implements matrix multiplication
        and returns relu of that
        """
        try:
            apply_relu = [0 if arb < 0 else arb for arb in data]
        except (TypeError, ValueError) as error:
            apply_relu = [0 if arb < 0 else arb for arb in data[0]]
        return np.array(apply_relu, dtype='float64').tolist()

    def sigmoid(self, data):
        """
        implements sigmoid function
        """
        try:
            return 1 / (1 + math.exp(-data))
        except:
            return 1/(1 + np.exp(-data))


    def dense_layer(self):
        """
        implements the actucal operations
        """
        result_final = np.zeros((len(self.input_data), self.units)).tolist()
        result_after_mat_mull = 0
        for inter_index, index in enumerate(range(0, len(self.input_data))):
            result_after_mat_mull = self.matrix_multiplication([self.input_data[index]])
            result_after_mat_mull = self.add_bias_terms(result_after_mat_mull)
            if self.activation == 'relu':
                result_after_mat_mull = self.relu(result_after_mat_mull)
            elif self.activation == 'sigmoid':
                result_after_mat_mull = self.sigmoid(result_after_mat_mull[0])
            else:
                print Fore.RED + "Warning: No activation function given"
            result_final[inter_index] = np.array(result_after_mat_mull, dtype='float64').tolist()
        return np.array(result_final, dtype='float64').tolist()



class MaxPoolingLayer(object):
    """
    implements maxpooling layer
    """
    def __init__(self, input_data, max_pool_length):
        self.input_data = input_data
        self.max_pool_length = max_pool_length

    def check_maxpool_length(self):
        """
        checks for length of the maxpool length
        """
        if self.max_pool_length > len(self.input_data):
            sys.exit(Fore.RED+"IndexError: Maxpool length should be less than the input array length" + Style.RESET_ALL)

    def max_pool_2d_array(self):
        """
        implements the max pooling on the two dimensional array / list
        """
        result_final = np.zeros((len(self.input_data)/self.max_pool_length, len(self.input_data[0]))).tolist()
        self.check_maxpool_length()
        intermediate_result = np.zeros((len(self.input_data[0]))).tolist()
        for index_value, index in enumerate(range(0, len(self.input_data), self.max_pool_length)):
            for columns in range(0, len(self.input_data[0])):
                result = [value[columns] for value in self.input_data[index:index+self.max_pool_length]]
                intermediate_result[columns] = max(result)
            result_final[index_value] = intermediate_result

        return np.array(result_final, dtype='float64').tolist()



class InitialCheckForShape(object):

    """
    checks for shape of weights and its corresponding layer outputs
    """
    def __init__(self, input_data, weights):
        self.input_data = input_data
        self.weights = weights

    def calculate_shape_conv_output(self, list_uks):
        """
        caluclates the shape of the output.
        Input argument is [ units, kernal_size, stride_length ]
        """
        if len(list_uks) == 3:
            output_shape = [len(self.input_data)/ list_uks[2], list_uks[1]]
            return output_shape
        else:
            print "Give all the paramters for convolution layer"

    def calculate_shape_dense_output(self, units):
        """
        calculates shape of dense layer
        """
        return [len(self.input_data), units]

    def calculate_shape_maxpool_output(self, max_pool_length):
        """
        calculates maxpool output
        """
        return [len(self.input_data)/max_pool_length, len(self.input_data[0])]

    def check_for_shape_alignment(self, dictionary_layers_units):
        """
        checks for shape alignment and returns True or False
        """
        get_num_layers = len(self.input_data)
        return get_num_layers



def unroll_the_architecture(arch_dict,layer_name, input_data,layer_index):
    """
    Initiates the neural network calculation as per the layer
    """
    try:
        if layer_name == "conv":
            cnn_rolled = ConvolutionalNeuralNetwork(units=int(arch_dict['units']),
                                                    input_data=input_data,
                                                    weights=np.array(WEIGHTS_VALUE[layer_index+1][0],
                                                                     dtype="float64").tolist(),
                                                    activation=arch_dict['activation'],
                                                    bias=np.array(WEIGHTS_VALUE[layer_index+1][1],
                                                                  dtype="float64").tolist())
            resut_value = cnn_rolled.start_window_moment(stride_length=int(arch_dict['stride_length']),
                                                         filter_size=int(arch_dict['filter_size']),
                                                         type_padding=arch_dict['type_padding'])
            # print np.array(resut_value).shape
            return resut_value
        elif layer_name == "maxpool":
            maxpool_rolled = MaxPoolingLayer(input_data=input_data,
                                             max_pool_length=int(arch_dict['max_pool_length']))
            return maxpool_rolled.max_pool_2d_array()
        elif layer_name == "dense":
            fully_connected = FullyConnectedLayer(units=int(arch_dict['units']),
                                                  input_data=input_data,
                                                  weights=np.array(WEIGHTS_VALUE[layer_index+1][0],
                                                                   dtype="float64").tolist(),
                                                  activation=arch_dict['activation'],
                                                  bias=np.array(WEIGHTS_VALUE[layer_index+1][1],
                                                                dtype="float64").tolist())
            return fully_connected.dense_layer()
        else:
            print Fore.RED + "ERROR: Invalid Layer used \n"
            sys.exit("try valid layer"+ Style.RESET_ALL)
    except:
        print Fore.RED + "MismatchError: Layer and Weights mismatch\n"
        sys.exit("Input the correct order"+Style.RESET_ALL)


def flag_for_downsampling(audiofilepath):
    """
    Implements three things.
    1. Reading the audio file that is being passed
    2. Applying the downsampling using the resampy library. We can also implement without using library but it is currently taking longer
    3. Generating the goertzel frequency components for each second of the audio file  [400Hz, 500Hz, 2000Hz, 2300Hz]
    3. Making the predictions for each second of the audio file
    """

    # Reading the audio file
    samplingrate_samples = downsampling_and_goertzel_filter.ReadAudioFile(audiofilepath).read_wav_file()
    try:
        samples_only = np.array([i[0] for i in samplingrate_samples[1]])
    except:
        samples_only = samplingrate_samples[1]

    # Downsampling the audio from any sampling rate to 8KHz for each second of the audio
    for seconds_index in range(0, samples_only.shape[0], samplingrate_samples[0])[:10]:
        down_library = downsampling_and_goertzel_filter.DownsampleUsingLibrary(samples_only[seconds_index:seconds_index+samplingrate_samples[0]], samplingrate_samples[0])
        resampled_using_lib = down_library.resample_using_resampy(8000)

        # Applying the goertzel filter on the downsampled audio
        for freqs in [400, 500, 2000, 2300]:
            if freqs == 400:
                goertzel = downsampling_and_goertzel_filter.GoertzelComponents(samples=resampled_using_lib,
                                                                               sample_rate=8000,
                                                                               target_frequency=freqs,
                                                                               number_samples=8000)
                goertzel_components = goertzel.goertzel_filter().tolist()
            else:
                goertzel = downsampling_and_goertzel_filter.GoertzelComponents(samples=resampled_using_lib,
                                                                               sample_rate=8000,
                                                                               target_frequency=freqs,
                                                                               number_samples=8000)
                goertzel_components_1 = goertzel.goertzel_filter().tolist()

                # Applying stacking on the axis =1
                for indices in range(len(goertzel_components)):
                    goertzel_components[indices].append(goertzel_components_1[indices][0])

        # Normalizing the data
        goertzel_components = goertzel_components / np.linalg.norm(goertzel_components)

        # Predicting using the goertzel model for each second
        for each_second in [goertzel_components]:
            start_time = time.time()
            for layer_index, layer_name in enumerate(ARCHITECTURE_CSV['layer_name'].values.tolist()):
                print Fore.GREEN + str(layer_index)+ " "+layer_name + " " +str(ARCHITECTURE_CSV.iloc[layer_index].to_dict()['units'])[:-2]+ Style.RESET_ALL
                if layer_index == 0 and layer_name != "EOF":
                    intermediate_result_values = unroll_the_architecture(ARCHITECTURE_CSV.iloc[layer_index].to_dict(),
                                                                         layer_name=layer_name,
                                                                         input_data=each_second,
                                                                         layer_index=layer_index
                                                                        )

                elif layer_name == "EOF":
                    print intermediate_result_values
                else:
                    intermediate_result_values = unroll_the_architecture(ARCHITECTURE_CSV.iloc[layer_index].to_dict(),
                                                                         layer_name=layer_name,
                                                                         input_data=intermediate_result_values,
                                                                         layer_index=layer_index
                                                                        )
            end_time = time.time() - start_time
            print Fore.YELLOW + "Time Elapsed: " +str(end_time) + Style.RESET_ALL


def predict_on_goertzelcomponents(ten_sec_data):
    """
    Predicts for a given goertzel components
    shape should of either
    (  1, 8000, 4 ) => 1 second  or
    ( 10, 8000, 4 ) => 10 seconds
    """
    for each_second in ten_sec_data:
        start_time = time.time()
        for layer_index, layer_name in enumerate(ARCHITECTURE_CSV['layer_name'].values.tolist()):
            print Fore.GREEN + str(layer_index)+ " "+layer_name + " " +str(ARCHITECTURE_CSV.iloc[layer_index].to_dict()['units'])[:-2]+ Style.RESET_ALL
            if layer_index == 0 and layer_name != "EOF":
                intermediate_result_values = unroll_the_architecture(ARCHITECTURE_CSV.iloc[layer_index].to_dict(),
                                                                     layer_name=layer_name,
                                                                     input_data=each_second,
                                                                     layer_index=layer_index
                                                                    )
                # print np.array(intermediate_result_values)

            elif layer_name == "EOF":
                print intermediate_result_values
            else:
                intermediate_result_values = unroll_the_architecture(ARCHITECTURE_CSV.iloc[layer_index].to_dict(),
                                                                     layer_name=layer_name,
                                                                     input_data=intermediate_result_values,
                                                                     layer_index=layer_index
                                                                    )
                # print np.array(intermediate_result_values)
        end_time = time.time() - start_time
        print Fore.YELLOW + "Time Elapsed: " +str(end_time) + Style.RESET_ALL

if __name__ == "__main__":
    print "start"
    '''
    If
    FLAG is set to 0, it takes audio file as input and starts the process of prediction from initial stage
    FLAG is set to 1, it takes generated goertzel frequency componets as input and gives the prediction
    '''
    FLAG = 0
    if FLAG == 0:
        flag_for_downsampling("-0RZOr28TNE-0.0-10.0.wav")

    else:
        predict_on_goertzelcomponents(TEN_SEC_INPUT_DATA)
