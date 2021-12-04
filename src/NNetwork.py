
from NNLayer import Layer
import sys
class NeuralNetwork:
    def __init__(self,Weights,Biases,Functions):
        self.Weights = Weights
        self.Biases = Biases
        self.Functions = Functions
        WeightsLength = len(Weights)
        BiasesLength = len(Biases)
        FunctionsLength = len(Functions)
        if WeightsLength == BiasesLength == FunctionsLength: #Function for checking lengths of list match
            pass
        else:
            print(Biases)
            raise ValueError("Lists are not off the same length")
        self.Layers = [] #Array of layer objects i.e [InputLayer,HiddenLayers,OutPutLayer]
        ##SOURCE: https://www.geeksforgeeks.org/python-pair-iteration-in-list/
        LayerInfo = zip(Weights,Biases,Functions) #Zip function will create a zip object this will allow us to create tuples pair each iteration of weights biases and functions essentially the information about each layer of the neural network
        LayerInfo = list(LayerInfo) # We need to convert zip from list to iterate over the tuples
        Count = 1
        for weight,bias,function in LayerInfo:
            if Count < len(LayerInfo): #Ensures we dont cause an IndexError
                if weight.shape[0] != LayerInfo[Count][0].shape[1]: #If the number of neurons does not = the number of weights output of the layer can not be used
                    print(weight.shape[0])
                    print(LayerInfo[Count][0].shape[1])
                    raise ValueError("Mismatch Matrix")
            layer = Layer(weight,bias,function)
            self.Layers.append(layer)
            Count+=1

    def getLayers(self):
        return self.Layers

    def propagate(self,NetworkInput):
        LayerInput = NetworkInput
        for layer in self.Layers:
            LayerInput = layer.forward(LayerInput) #We feed the output of an layer back every time until the end of the neural network
        return LayerInput

