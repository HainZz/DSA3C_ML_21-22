import numpy as np
import sys

#NOTES: Weights = [n1=[w1,w2,w3..,wX]n2=[w1,w2,w3,....,wX]...,nx=[w1,w2,...,wX]] 
# n = Neuron, w=Weight

class Layer:
    def __init__ (self,Weights,Biases,ActivationFunction):
        self.weights = np.copy(Weights) #Copies the Weights and Biases
        self.bias = np.copy(Biases)
        self.function = ActivationFunction
        neurons = self.weights.shape[0] #Gets the number of neurons bassed off the number of columns in matrix
        if neurons != len(self.bias): #If number of neurons + biases are not = then either a neuron dosent have a bias or vice versa
            raise ValueError("Incorrect Dimensions For Weights and Biases")

    def getMatrix(self):
        return self.weights

    def getBiasVector(self):
        return self.bias

    def getFunction(self):
        return self.function

    def forward(self,InputArray):
        x = np.dot(self.weights, InputArray) + self.bias #Here we can perform a dot product in order to work out
        self.function(x)
        return x

def main():
    pass