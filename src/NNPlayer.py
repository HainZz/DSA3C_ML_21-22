import numpy as np
import math 
from NNetwork import NeuralNetwork

class NNPlayer:
    def __init__(self,Weights,Biases,Functions):
        self.neuralnet = NeuralNetwork(Weights,Biases,Functions)

    def getSpecs():
        return (27,3) #Current spec 27 inputs repersenting each position on the board. The Value is the difference betweeen the players pips and the enemies at that position

    def SetFitness(self,FitnessScore):
        self.FitnessScore = FitnessScore

    def play(self,myBoard,OpponentBoard,myScore,OpponentScore,turn,GameLength,pips):
        MyNPBoard = np.array(myBoard)
        OpNPBoard = np.array(OpponentBoard)
        NetworkInput = MyNPBoard.ravel() - OpNPBoard.ravel() #This pip difference array is the difference between the players pips and the opponents pips at a given position on the board acts as the input into the neural network
        Result = self.neuralnet.propagate(NetworkInput) #Get result from the neural network this is currently an array of 3 numbers each repersenting a part of a full move
        Result = Result % 3
        Counter = 0
        for move in Result:
            Result[Counter] = int(math.floor(move))
            Counter += 1
        return list(map(int,Result))

    def getNN(self):
        return self.neuralnet