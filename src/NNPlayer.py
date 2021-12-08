import numpy as np
import math 
from NNetwork import NeuralNetwork

class NNPlayer:
    def __init__(self,Weights,Biases,Functions):
        self.neuralnet = NeuralNetwork(Weights,Biases,Functions)

    def getSpecs():
        return (27,9) #Current spec 27 inputs repersenting each position on the board. The Value is the difference betweeen the players pips and the enemies at that position

    def SetFitness(self,FitnessScore):
        self.FitnessScore = FitnessScore

    def play(self,myBoard,OpponentBoard,myScore,OpponentScore,turn,GameLength,pips):
        MyNPBoard = np.array(myBoard)
        OpNPBoard = np.array(OpponentBoard)
        NetworkInput = MyNPBoard.ravel() - OpNPBoard.ravel() #This pip difference array is the difference between the players pips and the opponents pips at a given position on the board acts as the input into the neural network
        Result = self.neuralnet.propagate(NetworkInput) #Get result from the neural network this is currently an array of 3 numbers each repersenting a part of a full move
        #Split output innto 3 seperate arrays get index for move based of higheest value in split list 
        Grid = np.argmax(Result[0:2])
        Row = np.argmax(Result[2:5])
        Column = np.argmax(Result[5:8])
        Move = [Grid,Row,Column]
        Move = list(map(int,Move))
        return Move

    def getNN(self):
        return self.neuralnet