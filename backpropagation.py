# -*- coding: utf-8 -*-
"""
Created on Mon May 14 01:26:36 2018

@author: Ranakrc
"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import numpy.random as r
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def activation_func(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(W, b, x):
    h = {1 : x}
    z = {}
    
    for l in range(1, len(W) + 1):
        # input is x for first layer otherwise output from last layer
        if l == 1:
            inp = x
        else:
            inp = h[l]
            
        # z^(l+1) = W^(l)*h^(l) + b^(l)
        z[l+1] = W[l].dot(inp) + b[l] 
        # h^(l) = f(z^(l))
        h[l+1] = activation_func(z[l+1])  
        
    return h, z


def hidden_layer(w_l, z_l, delta_next):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_next) * activation_func(z_l) * (1 - activation_func(z_l))


def output_layer(h_out, z_out, y):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return (h_out - y) * h_out * (1 - h_out)


def train(network, X, y):
    
    iteration = 300
    alpha = 0.5
    m = len(y)
    cost_list = []
    cnt = 0
    
    #initialization of W and b
    W = {}
    b = {}
    for l in range(1, len(network)):
        W[l] = r.random_sample((network[l], network[l-1]))
        b[l] = r.random_sample((network[l],))
    
    while cnt < iteration:
        cnt += 1
        
        #initialize Deltas of W and b
        del_W = {}
        del_b = {}
        for l in range(1, len(network)):
            del_W[l] = np.zeros((network[l], network[l-1]))
            del_b[l] = np.zeros((network[l],))
            
        cost = 0
        for i in range(len(y)):
            delta = {}
            h, z = forward_propagation(W, b, X[i, :])
            
            #backpropagating the errors
            for l in range(len(network), 0, -1):
                
                if l == len(network):
                    delta[l] = output_layer(h[l], z[l], y[i,:])
                    cost += np.linalg.norm((y[i,:]-h[l]))
                    
                else:
                    if l > 1:
                        delta[l] = hidden_layer(W[l], z[l], delta[l+1])
                        
                        # del_W[l] = del_W[l] + delta[l+1] * h[l]'
                        del_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis])) 
                        # del_b[l] = del_b[l] + delta[l+1]
                        del_b[l] += delta[l+1]
                    
        #accumulate the weight and bias update for each layer
        for l in range(len(network) - 1, 0, -1):
            W[l] += -alpha * ((1/m) * del_W[l])
            b[l] += -alpha * ((1/m) * del_b[l])
            
        cost = (1.0/m) * cost
        print(cost)
        cost_list.append(cost)
        
    return W, b, cost_list


def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(m):
        h, z = forward_propagation(W, b, X[i, :])
        y[i] = np.argmax(h[n_layers])
    return y

    
def main():

    r.seed(10)
    
    obs = 150;
    dims = 2;
    classes = 3;
    #Prep train and test data
    X_train = np.zeros((obs, dims))
    y_train = np.zeros(obs)
    X_test = np.zeros((obs, dims))
    y_test = np.zeros(obs)
    filename = "E:/Level 4, Term 2/CSE 474 - Pattern Recognition Sessional/Offline 1/train.csv"
    test = "E:/Level 4, Term 2/CSE 474 - Pattern Recognition Sessional/Offline 1/test.csv"
    data = pd.read_csv(filename, sep=",")
    data_test = pd.read_csv(test, sep=",")
    X_train[:,0] = data['col1']
    X_train[:,1] = data['col2']
    y_train = data['label']
    minimum = min(y_train)
    y_train = y_train - minimum
    X_test[:,0] = data_test['col1']
    X_test[:,1] = data_test['col2']
    y_test = data_test['label']
    y_test = y_test - minimum
    
    '''
    digits = load_digits()
    print(digits.data.shape) 
    plt.gray() 
    plt.matshow(digits.images[1]) 
    plt.show()    
    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    '''
    
    #convert y of training set to a vector
    y_vect_train = np.zeros((len(y_train), classes))
    for i in range(len(y_train)):
        y_vect_train[i, y_train[i]] = 1
    
    #convert y of test set to a vector
    y_vect_test = np.zeros((len(y_test), classes))
    for i in range(len(y_test)):
        y_vect_test[i, y_test[i]] = 1
    
    #y_train[0], y_vect_train[0]
    network = [2, 5, 10, 10, 5, 3]
    W, b, cost_list = train(network, X_train, y_vect_train)      
    
    plt.plot(cost_list)
    plt.ylabel('Average J')
    plt.xlabel('Iteration number')
    plt.show()

    y_pred = predict_y(W, b, X_test, len(network))
    print(accuracy_score(y_test, y_pred)*100)
    
if __name__== "__main__":
    main()

