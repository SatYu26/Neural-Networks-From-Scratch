import sys
import numpy as np
import matplotlib
from data import create_data

np.random.seed(0)

X = [[1.0, 2.0, 3.0, 2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]

X,y = create_data(100,3)


class Layer_Dense:
    def __init__(self, n_inputs , n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
        
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)
        
        

layer1 = Layer_Dense(2,5) # (size of input i.e. 'X' , any size we want for neurons)
# layer2 = Layer_Dense(5,2) # (size of output from layer1, any size we want for output)
Activation1 = Activation_ReLU()


layer1.forward(X)
# #print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)

Activation1.forward(layer1.output)
print(Activation1.output)
