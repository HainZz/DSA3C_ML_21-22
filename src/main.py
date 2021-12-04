import numpy as np
from numpy.lib.function_base import average
from numpy.random.mtrand import rand
from NNPlayer import NNPlayer
from CompromiseGame import DeterminedPlayer, GreedyPlayer, RandomPlayer,CompromiseGame, SmartGreedyPlayer
import pickle
import random
import statistics
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool,cpu_count, pool

##CONSTANT VARIABLES - Paramaters for the NN 
INPUT_NEURONS = 27
HLAYER_NEURONS = 15
OUTPUT_NEURONS = 9
CURRENT_GENERATION_FILE = "Generations/GENERATION_2.pkl"
CURRENT_BESTPLAYER_FILE = "BestPlayer/BP_V2.pkl"
POPULATION_SIZE = 1000
ELITE_PLAYER_PERCANTAGE = 10
NEURON_MUTATION_CHANCE = 0.40
GENERATIONS_COUNT = 100
PARENTS_IN_GENE_POOL = 200
#Ensures a more accurate fitness score so a lucky game != lucky fitness when actually genes are bad.
NO_OF_GAMES_FOR_FITNESS = 20
MAX = 5
MIN = -5
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

def play_game(playerA):
    playerB = RandomPlayer()
    PlayerAverageScore = []
    EnemyAverageScore = []
    for i in range(NO_OF_GAMES_FOR_FITNESS):
        game = CompromiseGame(playerA,playerB,30,10,"s")
        score = game.play()
        PlayerAverageScore.append(score[0])
        EnemyAverageScore.append(score[1])
    PlayerAverage = statistics.mean(PlayerAverageScore)
    EnemyAverage = statistics.mean(EnemyAverageScore)
    FitnessScore = PlayerAverage / EnemyAverage
    playerA.SetFitness(FitnessScore)
    return playerA

def main():
    ##DEVELOP INITIAL POPULATION
    InitialPopulation = []
    #InitialPopulation currently 100 random NN's paired with a fitness score which is esentially the difference between NN score and opponents score
    for x in range(POPULATION_SIZE):
        weights,biases,functions = InitializeNeuralNetworkWB()
        playerA = NNPlayer(weights,biases,functions)
        InitialPopulation.append(playerA) #Every object will have an fitness score associated with it within the generation.
    with Pool() as pool:
        InitialPopulation = pool.map(play_game,InitialPopulation)
    ##PICKLE Initial Generation
    #By saving the generations if we stop training we can continue training using the existing model rather than restarting every time.
    pickle_file = open(CURRENT_GENERATION_FILE,"wb")
    InitialPopulation.append(0)
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
    SelectedPopulation = [np.random.choice(generation,PARENTS_IN_GENE_POOL,p=SelectionChance)] #Here we select X parents from our objects with objects with higher fitness being selected.
    return SelectedPopulation

#Here we crossover the weights and biases of neurons from each layer ranging from 1-N-1 N=No of Neurons
def PerformOnePointCrossovers(Parent1Weight,Parent2Weight,Parent1Bias,Parent2Bias):
    #Gets the number of neurons within a layer this forms our maximum (-1 is for indexing)
    max = len(Parent1Weight)
    NoOfNeuronsToCross = np.random.randint(max)
    #Create a copy so we make sure we transfer original genes from Parent1Weights not new genes
    Parent1Weight[NoOfNeuronsToCross:max] = Parent2Weight[NoOfNeuronsToCross:max]
    Parent1Bias[NoOfNeuronsToCross:max] = Parent2Bias[NoOfNeuronsToCross:max]
    return Parent1Weight,Parent1Bias


def OnePointCrossover(generation,GenerationCount,NewGeneration):
    TotalPopulaton = POPULATION_SIZE - (ELITE_PLAYER_PERCANTAGE * POPULATION_SIZE / 100)
    cr = GenerationCount/GENERATIONS_COUNT
    c = cr * TotalPopulaton
    matingPool = BiasedRoluetteSelection(generation)
    matingPool = np.array(matingPool).flatten()
    Layers = []
    for i in range(round(c)): #Produce 90 children every crossover produces 2 children hence we do 45 crossovers.
        ChildFunctions = []
        Child1Weights = []
        Child1Bias = []
        SelectedParents = np.array([np.random.choice(matingPool,2)]).flatten() #Here the same parent avoids being selected twice NOTE: if somehow 2 of the same object are selected there is a chance that 2 genetically identical parents can be selecte
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
            Child1LayerWeight,Child1LayerBias = PerformOnePointCrossovers(Parent1Weight,Parent2Weight,Parent1Bias,Parent2Bias)
            Child1Weights.append(Child1LayerWeight)
            Child1Bias.append(Child1LayerBias)
        #Creation of children based of arrays from crossover
        Child1Object = NNPlayer(np.array(Child1Weights,dtype=object),np.array(Child1Bias,dtype=object),np.array(ChildFunctions,dtype=object))
        NewGeneration.append(Child1Object)
    return NewGeneration

##SOURCE: https://stackoverflow.com/questions/403421/how-to-sort-a-list-of-objects-based-on-an-attribute-of-the-objects
## This function aims to take N of the fittest parents and clone them into the next generation so we dont lose good genes. These cannot be mutated to ensure we dont lose a good gene with a bad mutation
def AddElite(generation):
    OrderedGeneration = sorted(generation, key=lambda x: x.FitnessScore, reverse=True) #Sort objects based off fitness score
    NoOFElite = ELITE_PLAYER_PERCANTAGE * POPULATION_SIZE / 100 #Get 20% of the population size
    NewGeneration = OrderedGeneration[:int(NoOFElite)] # Take N of the top objects from the OrderedGeneration this forms first N of our new generation
    return NewGeneration

#Currently every Bias + Weight within the children have a chance to mutate. 
#SOURCE: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiaqKulkcj0AhUCQEEAHWEoB0gQFnoECAoQAQ&url=https%3A%2F%2Fwww.mdpi.com%2F2078-2489%2F10%2F12%2F390%2Fpdf&usg=AOvVaw2uUvABCH2wTtCCVeDF8Vlp
def Mutation(Population,GenerationCount,generation):
    TotalPopulaton = POPULATION_SIZE - (ELITE_PLAYER_PERCANTAGE * POPULATION_SIZE / 100)
    MR = 1 - (GenerationCount/GENERATIONS_COUNT)
    NoOfMutatedPopulation = MR * TotalPopulaton
    mutation_pool = BiasedRoluetteSelection(generation)[0]
    for x in range(round(NoOfMutatedPopulation)):
        MutatedChild = np.random.choice(mutation_pool,1,replace=False) #Pick unique children to mutate 
        NeuralObject = MutatedChild[0].getNN()
        Weights = np.copy(NeuralObject.Weights)
        Biases = np.copy(NeuralObject.Biases)
        Functions = np.copy(NeuralObject.Functions)
        for LayerWeightIndex,Layer in enumerate(Weights):
            for NeuronIndex,Neuron in enumerate(Layer):
                for WeightIndex,Weight in enumerate(Neuron):
                    if random.random() < NEURON_MUTATION_CHANCE:
                        Weights[LayerWeightIndex][NeuronIndex][WeightIndex] = Weight - random.uniform(-1.0,1.0)
        for LayerBiasIndex,LayerBias in enumerate(Biases):
            for BiasIndex,bias in enumerate(LayerBias):
                if random.random() < NEURON_MUTATION_CHANCE:
                    Biases[LayerBiasIndex][BiasIndex] = bias - random.uniform(-1.0,1.0)
        MutatedOffSpring = NNPlayer(Weights,Biases,Functions)
        Population.append(MutatedOffSpring)
    return Population
## TODO 03/12/2021: Introduce Multi-Parent Crossover,Two Point Crossover,Universal Crossover,
#                   New Selection Method, Gaussian Mutation, Varying Crossover/Mutation Rates

def train():
    FitnessScoreY = []
    bpY = []
    GenerationCountX = []
    GenerationCount = 0
    for x in range(GENERATIONS_COUNT): #Train for a X generations.
        GenerationCount += 1
        Generation_File = open(CURRENT_GENERATION_FILE,"rb")
        generation = pickle.load(Generation_File)
        generation.pop()
        Generation_File.close()
        #Save Best Player From Prievous Generation
        OrderedGeneration = sorted(generation, key=lambda x: x.FitnessScore, reverse=True)
        BestPlayer = OrderedGeneration[:1][0]
        FitnessArray = []
        for player in OrderedGeneration:
            FitnessArray.append(player.FitnessScore)
        bpY.append(BestPlayer.FitnessScore)
        AF = statistics.mean(FitnessArray)
        FitnessScoreY.append(AF)
        GenerationCountX.append(GenerationCount)
        BestPlayer_File = open(CURRENT_BESTPLAYER_FILE,"wb")
        pickle.dump(BestPlayer,BestPlayer_File)
        BestPlayer_File.close()
        #Generate New Generation
        NewGeneration = AddElite(generation)
        NewGeneration = OnePointCrossover(generation,GenerationCount,NewGeneration)
        NewGeneration = Mutation(NewGeneration,GenerationCount,generation)
        #Assign Fitness Scores To Next Generation
        with Pool() as pool:
            NewGeneration = pool.map(play_game,NewGeneration)
        #Pickle Next Generation
        NewGeneration.append(GenerationCount)
        pickle_file = open(CURRENT_GENERATION_FILE,"wb")
        pickle.dump(NewGeneration,pickle_file)
        pickle_file.close()
        print(GenerationCount)
    #Benchmark Best Player
    test()
    Plot(GenerationCountX,FitnessScoreY,bpY)

def Plot(X,Y,bpy):
    plt.plot(X,Y,marker=".")
    plt.plot(X,bpy,marker="o")
    plt.xlabel("Generations")
    plt.ylabel("Average Fitness Score")
    plt.title("Average Fitness Score Over Generations")
    plt.legend([])
    plt.show()

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
        if ScoreDifference > 0:
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