"""
Implements the predictions of neural network output without using
any of the libraries. Its also modular and configurable
"""
import math
import pickle
import time
import numpy as np

def convolutional_layer(data, weights, kernel_size, strides, padding_type, units):
    """function goes here
    """
    if padding_type == "same":
        padding_length = calculate_padding_length(data, strides, kernel_size)
        data = apply_padding(data, padding_length)
        out_ind = 0
        final_result = [[0 for cols in range(units)] for rows in range(((len(data)-padding_length)/strides))]
        for index in range(0, (len(data) - padding_length), strides):
            result = matrix_mulitplication(data[index: index+kernel_size], weights[0], 0)
            result = add_bias(result, weights[1])
            result = activation_function(result, 'relu')
            final_result[out_ind] = result
            out_ind += 1

        return final_result



def add_bias(data, weights):
    """add bias
    """
    if (len(data) == len(weights) and type(data[0]) != list):
        result = [[0 for cols in range(len(weights))]for rows in range(len([data]))]
        result[0] = [sum(zip_values) for zip_values in zip(data, weights)]
        return result[0]
    elif len(data[0]) == len(weights):
        result = [[0 for cols in range(len(weights))]for rows in range(len(data))]
        for index in enumerate(data):
            result[index[0]] = [sum(zip_values) for zip_values in zip(index[1], weights)]
        return  result


def activation_function(data, type_activate):
    """
    check and implement the corresponding
    activation fucntion
    """
    if type_activate == 'relu':
        result = relu_function(data)
    elif type_activate == 'sigmoid':
        result = sigmoid(data)

    return result



def relu_function(data):
    """
    relu activation function
    """
    if type(data[0]) != list:
        result = [[0 for cols in range(len(data))]for rows in range(len([data]))]
        # for index in enumerate(data):
        result[0] = [0 if arb < 0 else arb for arb in data]
    elif type(data[0]) == list:
        result = [[0 for cols in range(len(data[0]))]for rows in range(len(data))]
        for index in enumerate(data):
            result[index[0]] = [0 if arb < 0 else arb for arb in index[1]]       
        return  result

    return result[0]

def max_pooling_layer(data, pool_length):

    """function goes here
    """
    result = np.zeros((len(data)/pool_length, len(data[0])))
    out_ind = 0

    for index in range(0, len(data), pool_length):
        for index_values in range(len(data[0])):
            result[out_ind][index_values] = max([len_data[index_values] for len_data in data[index:index+pool_length]])
        out_ind += 1
    return result.tolist()


def dense_layer(X, weights, type_activate):
    """
    function goes here
    """
    Y = weights[0]
    result = [[0 for cols in range(len(Y[0]))]for rows in range(len(X))]

    #perform Matrix Multiplication

    #iterate thorugh rows of X
    for i in range(len(X)):
       # iterate through columns of Y
        for j in range(len(Y[0])):
           # iterate through rows of Y
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]

    result = add_bias(result, weights[1])
    if type_activate != "nil":
        result = activation_function( result, type_activate)
    else:
        return result



    return result



def sigmoid(data):
    """
    opertaion goes here
    """
    return 1 / (1 + math.exp(-data))


def calculate_padding_length(data, strides, kernel_size):
    """
    calculation goes here
    """
    out_len = len(data)/(strides)
    padding_length = ((out_len - 1) * strides) - (len(data) - kernel_size)

    return padding_length



def apply_padding(data, padding_length):
    """
    process takes place here
    """
    data = data + [[0 for cols in range(len(data[0]))]for rows in range(padding_length)]
    return data


def matrix_mulitplication(X, Y, select):
    """
    multiplication goes here
    """
    # For convolution layer mulitplication
    if select == 0:
        length = Y[0]
        final_result = [0] * length[0]
        for index in enumerate(X):
            length = Y[index[0]]
            index = [index[1]]
            #Initialize the result list with zeros
            result = [[0] * len(length[0])]
            #perform Matrix Multiplication
            #iterate thorugh rows of X
            for i in range(len(index)):
               # iterate through columns of
                for j in range(len(length[0])):
                   # iterate through rows
                    for k in range(len(length)):
                        result[i][j] += index[i][k] * length[k][j]
            final_result = [sum(example) for example in zip(result[0], final_result)]

        return final_result



def keras_implementation(data, weights):
    """
    whole architecture
    """
    final_output = [0] * 10
    for each_second in enumerate(data):
        start_time = time.time()
        layer_output = max_pooling_layer(each_second[1], pool_length=4)
        layer_output = convolutional_layer(layer_output, weights[2], kernel_size=200, strides=100, padding_type="same", units=50)
        layer_output = convolutional_layer(layer_output, weights[3], kernel_size=4, strides=1, padding_type="same", units=50)
        layer_output = max_pooling_layer(layer_output, pool_length=20)
        layer_output = dense_layer(layer_output, weights[5], type_activate="relu")
        layer_output = dense_layer(layer_output, weights[6], type_activate="relu")
        layer_output = dense_layer(layer_output, weights[7], type_activate="nil")
        final_output[each_second[0]] = sigmoid(layer_output[0][0])
        end_time = time.time() - start_time
        print "Time Elapsed for one second prediction: ", end_time


    print final_output
    return final_output


if __name__ == "__main__":


    keras_implementation(data, weights)



