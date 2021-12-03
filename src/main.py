import numpy as np
from numpy.lib.function_base import average
from numpy.random.mtrand import rand
from NNPlayer import NNPlayer
from CompromiseGame import DeterminedPlayer, GreedyPlayer, RandomPlayer,CompromiseGame, SmartGreedyPlayer
import pickle
import random
import statistics

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
        playerA = NNPlayer(weights,biases,functions)
        InitialPopulation.append(playerA) #Every object will have an fitness score associated with it within the generation.
    play_game(InitialPopulation)
    ##PICKLE Initial Generation
    #By saving the generations if we stop training we can continue training using the existing model rather than restarting every time.
    pickle_file = open(CURRENT_GENERATION_FILE,"wb")
    pickle.dump(InitialPopulation,pickle_file)
    pickle_file.close()
    train()

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
    for i in range(45): #Produce 90 children every crossover produces 2 children hence we do 45 crossovers.
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
            Child1LayerWeight,Child2LayerWeight,Child1LayerBias,Child2LayerBias = Mutation(Child1LayerWeight,Child2LayerWeight,Child1LayerBias,Child2LayerBias)
            Child1Weights.append(Child1LayerWeight)
            Child2Weights.append(Child2LayerWeight)
            Child1Bias.append(Child1LayerBias)
            Child2Bias.append(Child2LayerBias)
        #Creation of children based of arrays from crossover
        Child1Object = NNPlayer(np.array(Child1Weights,dtype=object),np.array(Child1Bias,dtype=object),np.array(ChildFunctions,dtype=object))
        Child2Object = NNPlayer(np.array(Child2Weights,dtype=object),np.array(Child2Bias,dtype=object),np.array(ChildFunctions,dtype=object))
        NewPopulation.append(Child1Object)
        NewPopulation.append(Child2Object)
    return NewPopulation

##SOURCE: https://stackoverflow.com/questions/403421/how-to-sort-a-list-of-objects-based-on-an-attribute-of-the-objects
## This function aims to take N of the fittest parents and clone them into the next generation so we dont lose good genes. These cannot be mutated to ensure we dont lose a good gene with a bad mutation
def AddElite(generation,NewGeneration):
    OrderedGeneration = sorted(generation, key=lambda x: x.FitnessScore, reverse=True) #Sort objects based off fitness score
    NoOFElite = 10
    NewGeneration.extend(OrderedGeneration[:NoOFElite]) # Take N of the top objects from the OrderedGeneration
    return NewGeneration

#Currently every Bias + Weight within the children have a chance to mutate. 
def Mutation(Child1LayerWeight,Child2LayerWeight,Child1LayerBias,Child2LayerBias):
    WeightMutationChance = 0.05
    WeightMutationImpact = 0.3
    BiasMutationChance = 0.05 
    BiasMutationImpact = 1
    Weights = [Child1LayerWeight,Child2LayerWeight]
    Bias = [Child1LayerBias,Child2LayerBias]
    #Mutate Every Weight In Both Children
    c = 0 
    n = 0
    w = 0
    for child in Weights:
        n = 0
        for neuron in child:
            w = 0
            for weight in neuron:
                if random.random() < WeightMutationChance:
                    #50/50 chance of either + or - MutationImpact
                    if random.randint(0,1) == 1:
                        Weights[c][n][w] = weight + WeightMutationImpact
                    else:
                        Weights[c][n][w] = weight - WeightMutationImpact
                w += 1
            n += 1
        c += 1
    c = 0
    b = 0 
    for child in Bias:
        b = 0
        for bias in child:
            if random.random() < BiasMutationChance:
                if random.randint(0,1) == 1:
                    Bias[c][b] = bias + BiasMutationImpact
                else:
                    Bias[c][b] = bias - BiasMutationImpact
            b += 1
        c += 1
    return Weights[0], Weights[1], Bias[0], Bias[1]

## TODO 03/12/2021: Introduce Multi-Parent Crossover,Two Point Crossover,Universal Crossover,
#                   New Selection Method, Gaussian Mutation, Varying Crossover/Mutation Rates

def train():
    for x in range(500): #Train for a 500 generations.
        Generation_File = open(CURRENT_GENERATION_FILE,"rb")
        generation = pickle.load(Generation_File)
        Generation_File.close()
        #Save Best Player From Prievous Generation
        OrderedGeneration = sorted(generation, key=lambda x: x.FitnessScore, reverse=True)
        BestPlayer = OrderedGeneration[:1][0]
        BestPlayer_File = open(CURRENT_BESTPLAYER_FILE,"wb")
        pickle.dump(BestPlayer,BestPlayer_File)
        BestPlayer_File.close()
        #Generate New Generation
        NewGeneration = OnePointCrossover(generation)
        NewGeneration = AddElite(generation,NewGeneration)
        #Assign Fitness Scores To Next Generation
        play_game(NewGeneration)
        #Pickle Next Generation
        pickle_file = open(CURRENT_GENERATION_FILE,"wb")
        pickle.dump(NewGeneration,pickle_file)
        pickle_file.close()
    #Benchmark Best Player
    test()

#Designed as a benchmark for the Best Player
def test():
    #These arrays store results from the becnhmark
    RandomPlayerWL = [0,0]
    DeterminedPlayerWL = [0,0]
    GreedyPlayerWL = [0,0]
    SmartGreedyPlayerWL = [0,0]
    RandomPlayerAverageScore = []
    DeterminedPlayerAverageScore = []
    GreedyPlayerAverageScore = []
    SmartGreedyPlayerAverageScore = []
    #Get Best Player From Pickle File
    player_file = open(CURRENT_BESTPLAYER_FILE,"rb")
    playerA = pickle.load(player_file)
    player_file.close()
    #Play each player 1000 times collect W-L + scores NOTE: Potential Ties Are Counted As Losses 
    for i in range (1000):
        ##Play Against Random Player
        playerB = RandomPlayer()
        game = CompromiseGame(playerA,playerB,30,10,"s")
        score = game.play()
        ScoreDifference = score[0] - score[1]
        RandomPlayerAverageScore.append(ScoreDifference)
        if ScoreDifference <= 0:
            RandomPlayerWL[0] += 1
        else:
            RandomPlayerWL[1] += 1
        ##Play Against Determined Player
        playerB = DeterminedPlayer()
        game = CompromiseGame(playerA,playerB,30,10,"s")
        score = game.play()
        ScoreDifference = score[0] - score[1]
        DeterminedPlayerAverageScore.append(ScoreDifference)
        if ScoreDifference > 0:
            DeterminedPlayerWL[0] += 1
        else:
            DeterminedPlayerWL[1] += 1
        ##Play Against Greedy Player
        playerB = GreedyPlayer()
        game = CompromiseGame(playerA,playerB,30,10,"s")
        score = game.play()
        ScoreDifference = score[0] - score[1]
        GreedyPlayerAverageScore.append(ScoreDifference)
        if ScoreDifference > 0:
            GreedyPlayerWL[0] += 1
        else:
            GreedyPlayerWL[1] += 1
        ##Play Against SmartGreedy Player
        playerB = SmartGreedyPlayer()
        game = CompromiseGame(playerA,playerB,30,10,"s")
        score = game.play()
        ScoreDifference = score[0] - score[1]
        SmartGreedyPlayerAverageScore.append(ScoreDifference)
        if ScoreDifference > 0:
            SmartGreedyPlayerWL[0] += 1
        else:
            SmartGreedyPlayerWL[1] += 1
    PrintBenchmark(RandomPlayerWL,DeterminedPlayerWL,GreedyPlayerWL,SmartGreedyPlayerWL,RandomPlayerAverageScore,DeterminedPlayerAverageScore,GreedyPlayerAverageScore,DeterminedPlayerAverageScore)

def PrintBenchmark(RandomPlayerWL,DeterminedPlayerWL,GreedyPlayerWL,SmartGreedyPlayerWL,RandomPlayerAS,DeterminedPlayerAS,GreedyPlayerAS,SmartGreedyAS):
    print("Performance Against Random Player: \n Wins: ", RandomPlayerWL[0], " Loss: ", RandomPlayerWL[1], " WR: ", RandomPlayerWL[0]/1000 * 100 ,
    "\n AverageScore: ", statistics.mean(RandomPlayerAS))
    print("--------------------------------------------")
    print("Performance Against Determined Player: \n Wins: ", DeterminedPlayerWL[0], " Loss: ", DeterminedPlayerWL[1], " WR: ", DeterminedPlayerWL[0]/1000 * 100,
    "\n AverageScore: ", statistics.mean(DeterminedPlayerAS) )
    print("--------------------------------------------")
    print("Performance Against Greedy Player: \n Wins: ", GreedyPlayerWL[0], " Loss: ", GreedyPlayerWL[1], " WR: ", GreedyPlayerWL[0]/1000 * 100,
    "\n AverageScore: ", statistics.mean(GreedyPlayerAS) )
    print("--------------------------------------------")
    print("Performance Against Smart Greedy Player: \n Wins: ", SmartGreedyPlayerWL[0], " Loss: ", SmartGreedyPlayerWL[1], " WR: ", SmartGreedyPlayerWL[0]/1000*100,
    "\n AverageScore: ", statistics.mean(SmartGreedyAS))
    print("--------------------------------------------")


main()
