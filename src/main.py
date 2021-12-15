
import numpy as np
from NNPlayer import NNPlayer
from CompromiseGame import DeterminedPlayer, GreedyPlayer, RandomPlayer,CompromiseGame, SmartGreedyPlayer
import pickle
import statistics
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import repeat
#import Player1925723
import json

##CONSTANT VARIABLES - Paramaters for the NN 
INPUT_NEURONS = 27
HLAYER_NEURONS = 15
OUTPUT_NEURONS = 9
CURRENT_GENERATION_FILE = "Generations/TEST_GEN.pkl"
CURRENT_BESTPLAYER_FILE = "BestPlayer/TEST_BP.pkl"
#GA Parameters
POPULATION_SIZE = 1000
ELITE_PLAYER_PERCANTAGE = 10 #This passes the top 100 
NEURON_MUTATION_CHANCE = 0.40
GENERATIONS_COUNT = 5000
TOP_PARENT_PERCENTILE = 10
#Ensures a more accurate fitness score so a lucky game != lucky fitness when actually genes are bad.
#NO_OF_GAMES_FOR_FITNESS = 20 #Remeber this value is multiplied by 4 for each player
RANDOM_GAMES_PLAYED = 10
DETERMINED_GAMES_PLAYED = 10
GREEDY_GAMES_PLAYED = 10
SMART_GREEDY_GAMES_PLAYED = 10

def relu(Function_Input):
    return np.maximum(0.0,Function_Input)

##SOURCE: https://vidyasheela.com/post/leaky-relu-activation-function-with-python-code
## CURRENT ACTIVIATION FUNCTION
def leaky_relu(Function_Input):
    output = [max(0.01*value,value) for value in Function_Input]
    return np.array(output,dtype=float)


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
    functions = np.full((2),leaky_relu) #Fill our functions list with all RELU for now. We can experiment using a combination of functions later
    return weights,bias,functions

def play_game(playerA,GenerationCount):
    randomPlayer = RandomPlayer()
    determinedPlayer = DeterminedPlayer()
    greedyPlayer = GreedyPlayer()
    smartGreedy = SmartGreedyPlayer()    
    PlayerAverageScore = []
    EnemyAverageScore = []
    #Loop for X games and append scores to arrayy
    game = CompromiseGame(playerA,randomPlayer,30,10,"s")
    for x in range(RANDOM_GAMES_PLAYED):
        score = game.play()
        PlayerAverageScore.append(score[0])
        EnemyAverageScore.append(score[1])
    game = CompromiseGame(playerA,determinedPlayer,30,10,"s")
    for y in range(DETERMINED_GAMES_PLAYED):
        score = game.play()
        PlayerAverageScore.append(score[0])
        EnemyAverageScore.append(score[1])
    game = CompromiseGame(playerA,greedyPlayer,30,10,"s")
    for i in range(GREEDY_GAMES_PLAYED):
        score = game.play()
        PlayerAverageScore.append(score[0])
        EnemyAverageScore.append(score[1])
    game = CompromiseGame(playerA,smartGreedy,30,10,"s")
    for z in range(SMART_GREEDY_GAMES_PLAYED):
        score = game.play()
        PlayerAverageScore.append(score[0])
        EnemyAverageScore.append(score[1])
    #Get average of player + opponent score divide these to get the average fitness
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
        InitialPopulation = pool.starmap(play_game,zip(InitialPopulation,repeat(0)))
    ##PICKLE Initial Generation
    #By saving the generations if we stop training we can continue training using the existing model rather than restarting every time.
    pickle_file = open(CURRENT_GENERATION_FILE,"wb")
    InitialPopulation.append(0)
    pickle.dump(InitialPopulation,pickle_file)
    pickle_file.close()
    train()

##CURRENT SELECTION METHOD = https://en.wikipedia.org/wiki/Fitness_proportionate_selection
##NOTE: In This Method The Same Object Can Be Selected Twice
def BiasedRoluetteSelection(generation,NoOfParents):
    RawFitnessArray = []
    #Get 2 parents with higher fitness parents taking prio
    OrderedGeneration = getOrderedGeneration(generation)
    for chromosone in OrderedGeneration:
        RawFitnessArray.append(chromosone.FitnessScore)
    RawFitnessArray = np.array(RawFitnessArray) #Convert to NP array
    SelectionChance = RawFitnessArray / np.sum(RawFitnessArray)
    SelectedPopulation = [np.random.choice(OrderedGeneration,NoOfParents,p=SelectionChance,replace=False)] #Here we select X parents from our objects with objects with higher fitness being selected.
    return SelectedPopulation

##TODO - IMPLEMENT RANK SELECTION
def SteadyStateSelection(generation,NoOfParents):
    OrderedGeneration = getOrderedGeneration(generation)
    NoOFElite = TOP_PARENT_PERCENTILE * POPULATION_SIZE / 100
    SelectedParents = [np.random.choice(OrderedGeneration[:int(NoOFElite)],NoOfParents,replace=False)] #Instead of using a bias roulette we simply rank
    return SelectedParents

#Returns generations list ordered by fitness score from highest -> Lowest
def getOrderedGeneration(generation):
    OrderedGeneration = sorted(generation, key=lambda x: x.FitnessScore, reverse=True)
    return OrderedGeneration

#Here we crossover the weights and biases of neurons from each layer ranging from 1-N-1 N=No of Neurons
def PerformOnePointCrossovers(Parent1Weight,Parent2Weight,Parent1Bias,Parent2Bias):
    #Gets the number of neurons within a layer this forms our maximum (-1 is for indexing)
    max = len(Parent1Weight)
    NoOfNeuronsToCross = np.random.randint(max)
    #Exchange neurons forming the crossover process.
    Parent1Weight[NoOfNeuronsToCross:max] = Parent2Weight[NoOfNeuronsToCross:max]
    Parent1Bias[NoOfNeuronsToCross:max] = Parent2Bias[NoOfNeuronsToCross:max]
    return Parent1Weight,Parent1Bias


def OnePointCrossover(generation,SelectionFunction):
    NewGeneration = []
    TotalPopulaton = POPULATION_SIZE - (ELITE_PLAYER_PERCANTAGE * POPULATION_SIZE / 100)
    cr = 0.03
    c = cr * TotalPopulaton
    Layers = []
    #Get offspring made by crossover
    for i in range(round(c)): 
        ChildFunctions = []
        Child1Weights = []
        Child1Bias = []
        SelectedParents =  SelectionFunction(generation,2)#Call function to select parent
        #Get the layers of both parents.
        for parent in SelectedParents[0]:
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
    #Get offspring not made by crossover i.e Simply cloning a selected parent
    for i in range (int(TotalPopulaton-round(c))):
        SelectedParent = SelectionFunction(generation,1)
        NewGeneration.append(SelectedParent[0][0])
    return NewGeneration

##SOURCE: https://stackoverflow.com/questions/403421/how-to-sort-a-list-of-objects-based-on-an-attribute-of-the-objects
## This function aims to take N of the fittest parents and clone them into the next generation so we dont lose good genes. These cannot be mutated to ensure we dont lose a good gene with a bad mutation
def AddElite(generation,NewGeneration):
    OrderedGeneration = getOrderedGeneration(generation) #Sort objects based off fitness score
    NoOFElite = ELITE_PLAYER_PERCANTAGE * POPULATION_SIZE / 100 #Get 20% of the population size
    ElitePlayers = OrderedGeneration[:int(NoOFElite)] # Take N of the top objects from the OrderedGeneration this forms first N of our new generation
    NewGeneration = NewGeneration + ElitePlayers
    return NewGeneration

#Currently every Bias + Weight within the children have a chance to mutate. 
#SOURCE: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiaqKulkcj0AhUCQEEAHWEoB0gQFnoECAoQAQ&url=https%3A%2F%2Fwww.mdpi.com%2F2078-2489%2F10%2F12%2F390%2Fpdf&usg=AOvVaw2uUvABCH2wTtCCVeDF8Vlp
def Mutation(Population):
    TotalPopulaton = POPULATION_SIZE - (ELITE_PLAYER_PERCANTAGE * POPULATION_SIZE / 100)
    MR = 0.9
    NoOfMutatedPopulation = MR * TotalPopulaton
    ChildrenToMutate = np.random.choice(Population,size=int(NoOfMutatedPopulation),replace=False) #Ensure we dont pick the same chromosone to be mutated
    for child in ChildrenToMutate:
        NeuralObject = child.getNN()
        LayersObject = NeuralObject.getLayers()
        for layer in LayersObject:
            Matrix = layer.getMatrix()
            for NeuronCount,neuron in enumerate(Matrix):
                for WeightCount,weight in enumerate(neuron):
                    layer.weights[NeuronCount][WeightCount] = weight + np.random.uniform(-1.0,1.0)
            Biases = layer.getBiasVector()
            for BiasCount,bias in enumerate(Biases):
                layer.bias[BiasCount] = bias + np.random.uniform(-1.0,1.0)
    return Population

def train():
    FitnessScoreY = []
    bpY = []
    GenerationCountX = []
    for x in range(GENERATIONS_COUNT): #Train for a X generations.
        #Load prievous generation
        Generation_File = open(CURRENT_GENERATION_FILE,"rb")
        generation = pickle.load(Generation_File)
        currentGeneration = generation.pop()
        GenerationCount = currentGeneration + 1
        Generation_File.close()
        #Save Best Player From Prievous Generation + AF of poppulation for testing 
        OrderedGeneration = getOrderedGeneration(generation)
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
        NewGeneration = OnePointCrossover(generation,SteadyStateSelection)
        print(len(NewGeneration))
        NewGeneration = Mutation(NewGeneration)
        NewGeneration = AddElite(generation,NewGeneration)
        #Assign Fitness Scores To Next Generation
        with Pool() as pool:
            NewGeneration = pool.starmap(play_game,zip(NewGeneration,repeat(GenerationCount)))
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
    #playerA = Player1925723.NNPlayer()
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

def extract():
    playerDict = {}
    player_file = open(CURRENT_BESTPLAYER_FILE,"rb")
    player = pickle.load(player_file)
    player_file.close()
    NeuralObject=player.getNN()
    Layers=NeuralObject.getLayers()
    NewWeight = []
    NewBias = []
    for layer in Layers:
        NewWeight.append(layer.getMatrix().tolist())
        NewBias.append(layer.getBiasVector().tolist())
    playerDict["BestPlayer"] = {'Weights':NewWeight,'Bias':NewBias}
    with open('BestPlayer.json','w') as f:
        json.dump(playerDict,f)



#extract()
main()
#test()
