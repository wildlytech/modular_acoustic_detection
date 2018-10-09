import pickle
import numpy as np
import math
import time
import glob


def matrix_multiplication(X,Y, activation, bias):

    #Initialize the result list with zeros
    result = [ [0] * Y[0]]

    #perform Matrix Multiplication

    #iterate thorugh rows of X
    for i in range(len(X)):
       # iterate through columns of Y
       for j in range(len(Y[0])):
           # iterate through rows of Y
           for k in range(len(Y)):
               result[i][j] += X[i][k] * Y[k][j]


    # check for bias terms to add onto the results
    if type(bias)==np.ndarray:
        result = [[sum(zip_values) for zip_values in zip(result[0] ,bias )]]

        # check for activation function to apply and then do as per activation function
        if activation==1:
            result = [[0 if arb < 0 else arb for arb in result[0]]]

        elif activation==2:
            result = [[sigmoid(arb) for arb in result[0]]]

        else :
            pass

    # If no bisases then return the results
    else :
        pass

    return np.array(result,dtype ='float32').tolist()


def sigmoid(data):

    # apply sigmoid function and then  retun the results
    return 1 / (1 + math.exp(-data))


def samples_mul_weights_first_layer(data,weights,kernel_size,strides,num_units_1,activation,flag):
    #Initialize all the varibales

    # array_index iterates through the data [ samples ]
    array_index = 0
    # Its the index number of the loop
    index =0
    # index_weight iterates through the weights of that layer
    index_weight = 0
    #used for updating the window movement
    start= 0
    value = 0

    #check for the layer 1 or layer 2.
    # Flag =1 is layer 1.
    # data are the samples
    if flag == 1:
        # input for this layer will be one second = ( 8000,4 ) is the shape of data
        # Padding the data with appropiate length of zeros
        data = data + [[0 for cols in range(len(data[0]))]for rows in range(strides)]
        # calculate the length for which the nuber of iterations to take place
        length =(len(data))/(strides)
        # define the result list shape
        result_total1 = [[0 for x in range(num_units_1)] for y in range(length-1)]

    #Flag =2 is layer 2 implemetation
    else :
        #input for this layer will be of shape = ( 80, 100) is shape of data
        #padding the data with appropiate length of zeros
        data = data+[[0 for col in range(len(data[0])) ] for row in range(kernel_size-1)]
        # calculate the length for which the nuber of iterations to take place
        length =(len(data))/(strides)
        # define the result list shape
        result_total1 = [[0 for x in range(num_units_1)] for y in range(length-3)]


    # Actual implemenation of Convolution with window movement
    for index in range(length * kernel_size):

        # For the first compuation of loop or for first compuation after every window movement
        if index==0 or result_on_layer1 == 0:

            # convovling for the first element of data sample ie shape of first data sample will be ( 1,4 )
            # samples shape( 1,4 ) * weights shape ( 4,100 ) = output shape ( 1,100 )
            result_on_layer1 = matrix_multiplication([data[array_index]], weights[0][index_weight],0 , bias =0 )

        # Computes the convolving process till array_ index equals the kernel_size
        else:
            result_on_layer1 = [[sum(example) for example in zip(result_on_layer1[0], matrix_multiplication([data[array_index]] , weights[0][index_weight], 0, bias =0 )[0])]]

        #updating the variables after every convolution mulitplication
        index += 1
        array_index += 1
        index_weight += 1

        # updating the window movement if it is equal to kernel size ( in our case its 200 for layer1 and 4 for layer2  )
        if (array_index == int(start)+kernel_size) & (int(array_index) <= len(data)):

            #check if stride is more than
            if strides > 1:
                array_index -= strides
                index_weight = 0
                start += strides
            # strides equal to one ( which is default )
            else:
                array_index -= ( kernel_size - strides )
                index_weight = 0
                start += strides

            # check for the activation and then add the  biases, do as per activation function  and the update the result
            if activation == 'relu':
                result_on_layer1 = [[sum(zip_values) for zip_values in zip(result_on_layer1[0] , weights[1] )]]
                result_total1[value] = [0 if arb < 0 else arb for arb in result_on_layer1[0]]
                result_on_layer1 = 0
                value+=1
                # print value
            else :
                print 'Give the activatioon function'

            #check for shape of layer one ( In our case layer two must have ( 80, 100) output shape
            #This is for covolutional layer with strides more than one

            if flag==1:
                # return the values if the all the computation is done
                if value ==length -1:
                    return np.array(result_total1,dtype ='float32').tolist()
                else:
                    pass

            #check for shape of layer two ( In our case layer two must have ( 80, 100) output shape
            #This general for covolutional layerwith strides equal to one

            elif flag==0:
                # return the values if the all the computation is done
                if value == length - (kernel_size-1):
                    return np.array(result_total1,dtype ='float32').tolist()
                else:
                    pass
        # continue implementing concolution if window not equal to kernel size
        else:
            pass



def max_pooling_output(data_len_80_100):

    #Initialize the  variables
    column = 0
    max_pool_result = [0]*100

    #Iterate for as many times than length of the data
    for index in range(len(data_len_80_100[0])):

        #taking the each element of every samples of data at a time ( In our case 80 samples with each sample of 100 elements)
        result = [first_item_len_100[column] for first_item_len_100 in data_len_80_100]
        max_pool_result[column] = max(result)
        column+=1

    # return
    return max_pool_result



def layer1_predictions(data, weights):
    #Initialize the result
    i=0
    final_result = [0]*10
    data = data.tolist()

    #Iterate over evry second samples : ( It takes ( 1,8000,4)  as input for every iterations )
    for each_second in data:
        tic= time.time()
        layer1_output = samples_mul_weights_first_layer ( each_second, weights[1], kernel_size = 200, strides = 100, num_units_1 = 100, activation = 'relu', flag = 1 )
        layer1_output = samples_mul_weights_first_layer ( layer1_output, weights[2], kernel_size = 4, strides = 1, num_units_1 = 100, activation = 'relu', flag = 0 )
        layer1_output = max_pooling_output ( layer1_output )
        layer1_output = matrix_multiplication ( [ layer1_output ], weights[4][0] , 1, bias = weights[4][1])
        layer1_output = matrix_multiplication( layer1_output , weights[5][0] , 1 , bias = weights[5][1])
        layer1_output = matrix_multiplication( layer1_output , weights[6][0] , 2 , bias = weights[6][1])
        final_result[i] = layer1_output[0]
        i+=1
        toc = time.time() - tic
        print 'Time elapsed foe 1 second :', toc

    print 'Ouput after 6 layers :', max( final_result )
    return final_result
    # layer2_output



if __name__=='__main__':

    # calling for the predictions
    tic = time.time()
    final_output = layer1_predictions(data,weights )
    toc = time.time()
    print toc - tic
