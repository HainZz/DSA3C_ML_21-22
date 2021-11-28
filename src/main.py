import numpy as np
import sys
from NNPlayer import NNPlayer
from CompromiseGame import RandomPlayer,CompromiseGame

##CONSTANT VARIABLES - Paramaters for the NN 
INPUT_NEURONS = 27
HLAYER_NEURONS = 15
OUTPUT_NEURONS = 3

#CURRENT ACTIVIATION FUNCTION
def relu(Function_Input):
    return np.maximum(0.0,Function_Input)

#This function uses HE initialization
##SOURCE: https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
##SOURCE: https://stackoverflow.com/questions/22071987/generate-random-array-of-floats-between-a-range
#SOURCE: https://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
def InitializeNeuralNetworkWB():
    weights = np.array([np.random.uniform(HLAYER_NEURONS,HLAYER_NEURONS-1,size = (HLAYER_NEURONS,INPUT_NEURONS)),
    np.random.uniform(OUTPUT_NEURONS,OUTPUT_NEURONS-1,size = (OUTPUT_NEURONS,HLAYER_NEURONS))],dtype=object)
    n = 0
    for LayerWeights in weights:
        weights[n] = LayerWeights * np.sqrt(2/len(LayerWeights-1)) # * each weight with the sqrt of 2/size of layer - 1#
        n+= 1
    bias = np.array([np.zeros(HLAYER_NEURONS),np.zeros(OUTPUT_NEURONS)],dtype=object) #Initialize bias of 0
    functions = np.full((2),relu) #Fill our functions list with all RELU for now. We can experiment using a combination of functions later
    return weights,bias,functions

def main():
    weights,biases,functions = InitializeNeuralNetworkWB()
    playerA = NNPlayer(weights,biases,functions)
    playerB = RandomPlayer()
    game = CompromiseGame(playerA,playerB,30,10,"s")
    score = game.play()
    print(score)
    


main()