import numpy as np
import sys
from NNPlayer import NNPlayer
from CompromiseGame import RandomPlayer,CompromiseGame
import pickle

##CONSTANT VARIABLES - Paramaters for the NN 
INPUT_NEURONS = 27
HLAYER_NEURONS = 15
OUTPUT_NEURONS = 3
CURRENT_GENERATION_FILE = "Generations/GENERATION_V1.pkl"
CURRENT_BESTPLAYER_FILE = "BestPlayer/BP_V1.pkl"

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

def play_game(Population):
        playerB = RandomPlayer()
        for playerA in Population:
            game = CompromiseGame(playerA,playerB,30,10,"s")
            score = game.play()
            FitnessScore = score[0] / score[1]
            playerA.SetFitness(FitnessScore)

def main():
    ##DEVELOP INITIAL POPULATION
    InitialPopulation = []
    #InitialPopulation currently 100 random NN's paired with a fitness score which is esentially the difference between NN score and opponents score
    for x in range(100):
        weights,biases,functions = InitializeNeuralNetworkWB()
        print(biases)
        playerA = NNPlayer(weights,biases,functions)
        InitialPopulation.append(playerA) #Every object will have an fitness score associated with it within the generation.
    play_game(InitialPopulation)
    InitialPopulation.insert(0,0) #Array starts with 0. This number repersents the generation count. This is simply used for information regarding how many generations this model is on.
    ##PICKLE Initial Generation
    #By saving the generations if we stop training we can continue training using the existing model rather than restarting every time.
    pickle_file = open(CURRENT_GENERATION_FILE,"wb")
    pickle.dump(InitialPopulation,pickle_file)
    pickle_file.close()

##CURRENT SELECTION METHOD = https://en.wikipedia.org/wiki/Fitness_proportionate_selection
##NOTE: In This Method The Same Object Can Be Selected Twice
def BiasedRoluetteSelection(generation):
    RawFitnessArray = []
    for chromosone in generation:
        RawFitnessArray.append(chromosone.FitnessScore)
    RawFitnessArray = np.array(RawFitnessArray) #Convert to NP array
    SelectionChance = RawFitnessArray / np.sum(RawFitnessArray)
    SelectedPopulation = [np.random.choice(generation,20,p=SelectionChance)] #Here we select 20 parents from our objects with objects with higher fitness being selected.
    return SelectedPopulation

#Here we crossover the weights and biases of neurons from each layer ranging from 1-N-1 N=No of Neurons
def PerformOnePointCrossovers(Parent1Weight,Parent2Weight,Parent1Bias,Parent2Bias):
    #Gets the number of neurons within a layer this forms our maximum (-1 is for indexing)
    max = len(Parent1Weight)
    NoOfNeuronsToCross = np.random.randint(max)
    #Create a copy so we make sure we transfer original genes from Parent1Weights not new genes
    Parent1Copy = np.copy(Parent1Weight)
    Parent1Weight[NoOfNeuronsToCross:max] = Parent2Weight[NoOfNeuronsToCross:max]
    Parent2Weight[NoOfNeuronsToCross:max] = Parent1Copy[NoOfNeuronsToCross:max]
    Parent1BiasCopy = np.copy(Parent1Bias)
    Parent1Bias[NoOfNeuronsToCross:max] = Parent2Bias[NoOfNeuronsToCross:max]
    Parent2Bias[NoOfNeuronsToCross:max] = Parent1BiasCopy[NoOfNeuronsToCross:max]
    return Parent1Weight,Parent2Weight,Parent1Bias,Parent2Bias


def OnePointCrossover(generation):
    matingPool = BiasedRoluetteSelection(generation)
    matingPool = np.array(matingPool).flatten()
    NewPopulation = []
    Layers = []
    for i in range(30): #Produce 60 children every crossover produces 2 children hence we do 30 crossovers.
        ChildFunctions = []
        Child1Weights = []
        Child2Weights = []
        Child1Bias = []
        Child2Bias = []
        SelectedParents = np.array([np.random.choice(matingPool,2,False)]).flatten() #Here the same parent avoids being selected twice NOTE: if somehow 2 of the same object are selected there is a chance that 2 genetically identical parents can be selecte
        #Get the layers of both parents.
        for parent in SelectedParents:
            NeuralObject = parent.getNN()
            LayerObjects = NeuralObject.getLayers()
            Layers.append(LayerObjects)
        #Get the weights and biases for child based off crossover from 2 parents.
        for i in range (len(Layers[0])):
            #Stick the weights in a 2d array [0] is weights/bias of first parent [1] is W/B values for second parent
            Parent1Weight = Layers[0][i].getMatrix()
            Parent2Weight = Layers[1][i].getMatrix()
            Parent1Bias = Layers[0][i].getBiasVector()
            Parent2Bias = Layers[1][i].getBiasVector()
            #Get Function From Single Parent ATM functions match between NN's
            ChildFunctions.append(Layers[0][i].getFunction())
            Child1LayerWeight,Child2LayerWeight,Child1LayerBias,Child2LayerBias = PerformOnePointCrossovers(Parent1Weight,Parent2Weight,Parent1Bias,Parent2Bias)
            Child1Weights.append(Child1LayerWeight)
            Child2Weights.append(Child2LayerWeight)
            Child1Bias.append(Child1LayerBias)
            Child2Bias.append(Child2LayerBias)
        print(np.array(Child1Bias,dtype=object))
        #Creation of children based of arrays from crossover
        Child1Object = NNPlayer(np.array(Child1Weights),np.array(Child1Bias),np.array(ChildFunctions))
        Child2Object = NNPlayer(np.array(Child2Weights),np.array(Child2Bias),np.array(ChildFunctions))

def Mutation():
    pass

def train():
    for x in range(100): #Train for a 100 generations.
        Generation_File = open(CURRENT_GENERATION_FILE,"rb")
        generation = pickle.load(Generation_File)
        Generation_Number = generation.pop(0) #Pop the number off the array. This will be added back later when we overwrite the file with a new generation.
        OnePointCrossover(generation)
        sys.exit(0)

def test():
    pass


#test() 
#train()
main()