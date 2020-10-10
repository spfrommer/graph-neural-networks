# 2018/12/03~
# Fernando Gama, fgama@seas.upenn.edu
# Luana Ruiz, rubruiz@seas.upenn.edu

# Simulate the source localization problem. We have a graph, and we observe a
# signal defined on top of this graph. This signal is assumed to represent the
# diffusion of a rumor. The rumor is observed after being diffused for an
# unknown amount of time. The objective is to determine which is the node (or 
# the community) that started the rumor.

# Outputs:
# - Text file with all the hyperparameters selected for the run and the 
#   corresponding results (hyperparameters.txt)
# - Pickle file with the random seeds of both torch and numpy for accurate
#   reproduction of results (randomSeedUsed.pkl)
# - The parameters of the trained models, for both the Best and the Last
#   instance of each model (savedModels/)
# - The figures of loss and evaluation through the training iterations for
#   each model (figs/ and trainVars/)
# - If selected, logs in tensorboardX certain useful training variables

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import pickle
import datetime
from copy import deepcopy

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
import Modules.architectures as archit
import Modules.model as model
import Modules.training as training
import Modules.evaluation as evaluation
import Modules.loss as loss

#\\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

import pdb

# Start measuring time
startRunTime = datetime.datetime.now()

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

# TODO: implement geometric
graphType = 'geometric' # Type of graph: 'SBM', 'SmallWorld', 'geometric'

thisFilename = 'subspaces' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + graphType + '-' + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters are results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########

useGPU = True # If true, and GPU is available, use it.

nTrain = 8000 # Number of training samples
nValid = int(0.025 * nTrain) # Number of validation samples
nTest = 200 # Number of testing samples
tMax = 25 # Maximum number of diffusion times (A^t for t < tMax)

nDataRealizations = 3 # Number of data realizations
nGraphRealizations = 3 # Number of graph realizations
nClasses = 5 # Number of source nodes to select

nNodes = 100 # Number of nodes
graphOptions = {} # Dictionary of options to pass to the createGraph function
if graphType == 'SBM':
    graphOptions['nCommunities'] = nClasses # Number of communities
    graphOptions['probIntra'] = 0.8 # Intracommunity probability
    graphOptions['probInter'] = 0.2 # Intercommunity probability
elif graphType == 'SmallWorld':
    graphOptions['probEdge'] = 0.5 # Edge probability
    graphOptions['probRewiring'] = 0.1 # Probability of rewiring
elif graphType == 'geometric':
    graphOptions['pos'] = [None]
    graphOptions['kernelType'] = 'exponential'
    graphOptions['sparseType'] = 'NN'
    graphOptions['sparseParam'] = 5

#\\\ Save values:
writeVarValues(varsFile, {'nNodes': nNodes, 'graphType': graphType})
writeVarValues(varsFile, graphOptions)
writeVarValues(varsFile, {'nTrain': nTrain,
                          'nValid': nValid,
                          'nTest': nTest,
                          'tMax': tMax,
                          'nDataRealizations':nDataRealizations,
                          'nGraphRealizations': nGraphRealizations,
                          'nClasses': nClasses,
                          'useGPU': useGPU})

############
# TRAINING #
############

#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.001 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.MSELoss
    # it into the NLL, so we don't have to apply the softmax ourselves.

#\\\ Overall training options
nEpochs = 40 # Number of epochs
batchSize = 100 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 20 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'optimAlg': optimAlg,
                'learningRate': learningRate,
                'beta1': beta1,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# Select desired architectures
doFilter = True
doLocalGNN = True

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. Do not forget to add the name of the architecture
# to modelList.

# If the model dictionary is called 'model' + name, then it can be
# picked up immediately later on, and there's no need to recode anything after
# the section 'Setup' (except for setting the number of nodes in the 'N' 
# variable after it has been coded).

# The name of the keys in the model dictionary have to be the same
# as the names of the variables in the architecture call, because they will
# be called by unpacking the dictionary.

modelList = []

#\\\\\\\\\\\\\\\\\\\\\
#\\\ LOCAL GNN \\\
#\\\\\\\\\\\\\\\\\\\\\

if doFilter:
    
    #\\\ Basic parameters for all the Local GNN architectures
    
    modelFilter = {}
    modelFilter['name'] = 'Filter' # To be modified later on depending on the
        # specific ordering selected
    modelFilter['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
                                     
    #\\\ ARCHITECTURE
        
    # Select architectural nn.Module to use
    modelFilter['archit'] = archit.LocalGNN
    # Graph convolutional layers
    modelFilter['dimNodeSignals'] = [1, 1] # Number of features per layer
    modelFilter['nFilterTaps'] = [3] # Number of filter taps
    modelFilter['bias'] = True # Include bias
    # Nonlinearity
    modelFilter['nonlinearity'] = gml.NoActivation 
    # Pooling
    modelFilter['poolingFunction'] = gml.NoPool # Summarizing function
    modelFilter['nSelectedNodes'] = [nNodes]
    # modelFilter['nSelectedNodes'] = None 
    modelFilter['poolingSize'] = [1] # poolingSize-hop neighborhood that
                                        # is affected by the summary
    # Readout layer
    # modelFilter['dimReadout'] = []
    modelFilter['dimReadout'] = [1]
    # Graph Structure
    modelFilter['GSO'] = None # To be determined later on, based on data
    modelFilter['order'] = None
    
    #\\\ TRAINER

    modelFilter['trainer'] = training.Trainer
    
    #\\\ EVALUATOR
    
    modelFilter['evaluator'] = evaluation.evaluate

    modelList += [modelFilter['name']]


if doLocalGNN:
    
    #\\\ Basic parameters for all the Local GNN architectures
    
    modelLocalGNN = {}
    modelLocalGNN['name'] = 'LocalGNN' # To be modified later on depending on the
        # specific ordering selected
    modelLocalGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) \
                                     else 'cpu'
                                     
    #\\\ ARCHITECTURE
        
    # Select architectural nn.Module to use
    modelLocalGNN['archit'] = archit.LocalGNN
    # Graph convolutional layers
    modelLocalGNN['dimNodeSignals'] = [1, 32, 32] # Number of features per layer
    modelLocalGNN['nFilterTaps'] = [3, 3] # Number of filter taps
    modelLocalGNN['bias'] = True # Include bias
    # Nonlinearity
    modelLocalGNN['nonlinearity'] = nn.Tanh
    # Pooling
    modelLocalGNN['poolingFunction'] = gml.NoPool # Summarizing function
    modelLocalGNN['nSelectedNodes'] = [nNodes, nNodes]
    # modelLocalGNN['nSelectedNodes'] = None 
    modelLocalGNN['poolingSize'] = [1, 1] # poolingSize-hop neighborhood that
                                        # is affected by the summary
    # Readout layer
    # modelLocalGNN['dimReadout'] = []
    modelLocalGNN['dimReadout'] = [1]
    # Graph Structure
    modelLocalGNN['GSO'] = None # To be determined later on, based on data
    modelLocalGNN['order'] = None
    
    #\\\ TRAINER

    modelLocalGNN['trainer'] = training.Trainer
    
    #\\\ EVALUATOR
    
    modelLocalGNN['evaluator'] = evaluation.evaluate

    modelList += [modelLocalGNN['name']]

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 0 # After how many training steps, print the partial results
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 10 # How many validation steps in between those shown,
    # same as above.
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doLogging': doLogging,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'saveDir': saveDir,
                'printInterval': printInterval,
                'figSize': figSize,
                'lineWidth': lineWidth,
                'markerShape': markerShape,
                'markerSize': markerSize})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Determine processing unit:
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()

#\\\ Notify of processing units
if doPrint:
    print("Selected devices:")
    for thisModel in modelList:
        modelDict = eval('model' + thisModel)
        print("\t%s: %s" % (thisModel, modelDict['device']))

#\\\ Logging options
if doLogging:
    from Utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')
    
#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each of the trained models.
# It basically is a dictionary, containing a list. The key of the
# dictionary determines the model, then the first list index determines
# which split realization. Then, this will be converted to numpy to compute
# mean and standard deviation (across the split dimension).
costBest = {} # Cost for the best model (Evaluation cost: Error rate)
costLast = {} # Cost for the last model
for thisModel in modelList: # Create an element for each split realization,
    costBest[thisModel] = [None] * nGraphRealizations
    costLast[thisModel] = [None] * nGraphRealizations

if doFigs:
    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # for each data split.
    # Each data split, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    costTrain = {}
    lossValid = {}
    costValid = {}
    # Initialize the splits dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] * nGraphRealizations
        costTrain[thisModel] = [None] * nGraphRealizations
        lossValid[thisModel] = [None] * nGraphRealizations
        costValid[thisModel] = [None] * nGraphRealizations


####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of this options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doLogging:
    trainingOptions['logger'] = logger
if doSaveVars:
    trainingOptions['saveDir'] = saveDir
if doPrint:
    trainingOptions['printInterval'] = printInterval
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval

# And in case each model has specific training options, then we create a 
# separate dictionary per model.

trainingOptsPerModel= {}

#%%##################################################################
#                                                                   #
#                    GRAPH REALIZATION                              #
#                                                                   #
#####################################################################

# Start generating a new graph for each of the number of graph realizations that
# we previously specified.

# Unless it's the Facebook graph, which is fixed

# Load the graph and select the source nodes

for graph in range(nGraphRealizations):

    # The accBest and accLast variables, for each model, have a list with a
    # total number of elements equal to the number of graphs we will generate
    # Now, for each graph, we have multiple data realization, so we want, for
    # each graph, to create a list to hold each of those values
    for thisModel in modelList:
        costBest[thisModel][graph] = []
        costLast[thisModel][graph] = []
        
        lossTrain[thisModel][graph] = []
        costTrain[thisModel][graph] = []
        lossValid[thisModel][graph] = []
        costValid[thisModel][graph] = []       

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    # Create graph
    G = graphTools.Graph(graphType, nNodes, graphOptions)
    G.computeGFT() # Compute the eigendecomposition of the stored GSO

    # We have now created the graph and selected the source nodes on that graph.
    # So now we proceed to generate random data realizations, different
    # realizations of diffusion processes.

    for realization in range(nDataRealizations):

        ############
        # DATASETS #
        ############

        #   Now that we have the list of nodes we are using as sources, then we
        #   can go ahead and generate the datasets.
        # data = Utils.dataTools.SourceLocalization(G, nTrain, nValid, nTest,
                                                  # sourceNodes, tMax = tMax)
        data = Utils.dataTools.Wireless(G, nTrain, nValid, nTest)
        data.astype(torch.float64)
        #data.to(device)
        data.expandDims() # Data are just graph signals, but the architectures 
            # require that the input signals are of the form B x F x N, so we
            # need to expand the middle dimensions to convert them from B x N 
            # to B x 1 x N

        #%%##################################################################
        #                                                                   #
        #                    MODELS INITIALIZATION                          #
        #                                                                   #
        #####################################################################
        
        # This is the dictionary where we store the models (in a model.Model
        # class, that is then passed to training).
        modelsGNN = {}
    
        # If a new model is to be created, it should be called for here.
        
        if doPrint:
            print("Model initialization...", flush = True)
        
        for thisModel in modelList:
            
            # Get the corresponding parameter dictionary
            modelDict = deepcopy(eval('model' + thisModel))
            # and training options
            trainingOptsPerModel[thisModel] = deepcopy(trainingOptions)
            
            # Now, this dictionary has all the hyperparameters that we need to
            # pass to the architecture function, but it also has other keys
            # that belong to the more general model (like 'name' or 'device'),
            # so we need to extract them and save them in seperate variables
            # for future use.
            thisName = modelDict.pop('name')
            callArchit = modelDict.pop('archit')
            thisDevice = modelDict.pop('device')
            thisTrainer = modelDict.pop('trainer')
            thisEvaluator = modelDict.pop('evaluator')
            
            # If more than one graph or data realization is going to be 
            # carried out, we are going to store all of thos models
            # separately, so that any of them can be brought back and
            # studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
                
            if doPrint:
                print("\tInitializing %s..." % thisName,
                      end = ' ',flush = True)
                
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisOptimAlg = optimAlg
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ GSO
            # The coarsening technique is defined for the normalized and
            # rescaled Laplacian, whereas for the other ones we use the
            # normalized adjacency
            if 'crs' in thisModel:
                L = graphTools.normalizeLaplacian(G.L)
                EL, VL = graphTools.computeGFT(L, order = 'increasing')
                S = 2*L/np.max(np.real(EL)) - np.eye(nNodes)
            else:
                S = G.S.copy()/np.max(np.real(G.E))
                
            modelDict['GSO'] = S
            
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = callArchit(**modelDict)
            
            #############
            # OPTIMIZER #
            #############
    
            if thisOptimAlg == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate,
                                       betas = (beta1, beta2))
            elif thisOptimAlg == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(),
                                      lr = learningRate)
            elif thisOptimAlg == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            # Initialize the loss function
            thisLossFunction = lossFunction()
    
            #########
            # MODEL #
            #########
    
            # Create the model
            modelCreated = model.Model(thisArchit,
                                       thisLossFunction,
                                       thisOptim,
                                       thisTrainer,
                                       thisEvaluator,
                                       thisDevice,
                                       thisName,
                                       saveDir)
    
            # Store it
            modelsGNN[thisName] = modelCreated
    
            # Write the main hyperparameters
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisOptimizationAlgorithm': thisOptimAlg,
                            'thisTrainer': thisTrainer,
                            'thisEvaluator': thisEvaluator,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
            if doPrint:
                print("OK")
                
        if doPrint:
            print("Model initialization... COMPLETE")
    
        #%%##################################################################
        #                                                                   #
        #                    TRAINING                                       #
        #                                                                   #
        #####################################################################

        print("")
        
        # We train each model separately
    
        for thisModel in modelsGNN.keys():
            
            if doPrint:
                print("Training model %s..." % thisModel)
             
            # Remember that modelsGNN.keys() has the split numbering as well as 
            # the name, while modelList has only the name. So we need to map 
            # the specific model for this specific split with the actual model
            # name, since there are several variables that are indexed by the
            # model name (for instance, the training options, or the
            # dictionaries saving the loss values)
            for m in modelList:
                if m in thisModel:
                    modelName = m
        
            # Identify the specific graph and data realizations at training time
            if nGraphRealizations > 1:
                trainingOptions['graphNo'] = graph
            if nDataRealizations > 1:
                trainingOptions['realizationNo'] = realization
            
            # Train the model
            thisTrainVars = modelsGNN[thisModel].train(data,
                                                       nEpochs,
                                                       batchSize,
                                                       **trainingOptsPerModel[modelName])
    
            if doFigs:
            # Find which model to save the results (when having multiple
            # realizations)
                lossTrain[modelName][graph] += [thisTrainVars['lossTrain']]
                costTrain[modelName][graph] += [thisTrainVars['costTrain']]
                lossValid[modelName][graph] += [thisTrainVars['lossValid']]
                costValid[modelName][graph] += [thisTrainVars['costValid']]
                        
        # And we also need to save 'nBatch' but is the same for all models, so
        if doFigs:
            nBatches = thisTrainVars['nBatches']

        #%%##################################################################
        #                                                                   #
        #                    EVALUATION                                     #
        #                                                                   #
        #####################################################################
        
        # Now that the model has been trained, we evaluate them on the test
        # samples.
    
        # We have two versions of each model to evaluate: the one obtained
        # at the best result of the validation step, and the last trained model.
    
        if doPrint:
            print("\nTotal testing error rate", end = '', flush = True)
            if nGraphRealizations > 1 or nDataRealizations > 1:
                print(" (", end = '', flush = True)
                if nGraphRealizations > 1:
                    print("Graph %02d" % graph, end = '', flush = True)
                    if nDataRealizations > 1:
                        print(", ", end = '', flush = True)
                if nDataRealizations > 1:
                    print("Realization %02d" % realization, end = '',
                          flush = True)
                print(")", end = '', flush = True)
            print(":", flush = True)
            
    
        for thisModel in modelsGNN.keys():
            
            # Same as before, separate the model name from the data or graph
            # realization number
            for m in modelList:
                if m in thisModel:
                    modelName = m
    
            # Evaluate the model
            thisEvalVars = modelsGNN[thisModel].evaluate(data)
            
            # Save the outputs
            thisCostBest = thisEvalVars['costBest']
            thisCostLast = thisEvalVars['costLast']
            
            # Write values
            writeVarValues(varsFile,
                           {'costBest%s' % thisModel: thisCostBest,
                            'costLast%s' % thisModel: thisCostLast})
    
            # Now check which is the model being trained
            costBest[modelName][graph] += [thisCostBest]
            costLast[modelName][graph] += [thisCostLast]
            # This is so that we can later compute a total accuracy with
            # the corresponding error.
            
            if doPrint:
                print("\t%s: %6.4f%% [Best] %6.4f%% [Last]" % (thisModel,
                                                               thisCostBest,
                                                               thisCostLast))

############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanCostBestPerGraph = {} # Compute the mean accuracy (best) across all
    # realizations data realizations of a graph
meanCostLastPerGraph = {} # Compute the mean accuracy (last) across all
    # realizations data realizations of a graph
meanCostBest = {} # Mean across graphs (after having averaged across data
    # realizations)
meanCostLast = {} # Mean across graphs
stdDevCostBest = {} # Standard deviation across graphs
stdDevCostLast = {} # Standard deviation across graphs

if doPrint:
    print("\nFinal evaluations (%02d graphs, %02d realizations)" % (
            nGraphRealizations, nDataRealizations))

for thisModel in modelList:
    # Convert the lists into a nGraphRealizations x nDataRealizations matrix
    costBest[thisModel] = np.array(costBest[thisModel])
    costLast[thisModel] = np.array(costLast[thisModel])
    
    if nGraphRealizations == 1 or nDataRealizations == 1:
        meanCostBestPerGraph[thisModel] = np.squeeze(costBest[thisModel])
        meanCostLastPerGraph[thisModel] = np.squeeze(costLast[thisModel])
    else:
        # Compute the mean (across realizations for a given graph)
        meanCostBestPerGraph[thisModel] = np.mean(costBest[thisModel], axis = 1)
        meanCostLastPerGraph[thisModel] = np.mean(costLast[thisModel], axis = 1)

    # And now compute the statistics (across graphs)
    meanCostBest[thisModel] = np.mean(meanCostBestPerGraph[thisModel])
    meanCostLast[thisModel] = np.mean(meanCostLastPerGraph[thisModel])
    stdDevCostBest[thisModel] = np.std(meanCostBestPerGraph[thisModel])
    stdDevCostLast[thisModel] = np.std(meanCostLastPerGraph[thisModel])

    # And print it:
    if doPrint:
        print("\t%s: %6.4f%% (+-%6.4f%%) [Best] %6.4f%% (+-%6.4f%%) [Last]" % (
                thisModel,
                meanCostBest[thisModel],
                stdDevCostBest[thisModel],
                meanCostLast[thisModel],
                stdDevCostLast[thisModel]))

    # Save values
    writeVarValues(varsFile,
               {'meanCostBest%s' % thisModel: meanCostBest[thisModel],
                'stdDevCostBest%s' % thisModel: stdDevCostBest[thisModel],
                'meanCostLast%s' % thisModel: meanCostLast[thisModel],
                'stdDevCostLast%s' % thisModel : stdDevCostLast[thisModel]})
    
with open(varsFile, 'a+') as file:
    file.write("Final evaluations (%02d graphs, %02d realizations)\n" % (
            nGraphRealizations, nDataRealizations))
    for thisModel in modelList:
        file.write("\t%s: %6.4f%% (+-%6.4f%%) [Best] %6.4f%% (+-%6.4f%%) [Last]\n" % (
                   thisModel,
                   meanCostBest[thisModel],
                   stdDevCostBest[thisModel],
                   meanCostLast[thisModel],
                   stdDevCostLast[thisModel]))
    file.write('\n')

#%%##################################################################
#                                                                   #
#                    PLOT                                           #
#                                                                   #
#####################################################################

# Finally, we might want to plot several quantities of interest

if doFigs and doSaveVars:
    
    ###################
    # DATA PROCESSING #
    ###################
    
    #\\\ FIGURES DIRECTORY:
    saveDirFigs = os.path.join(saveDir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(saveDirFigs):
        os.makedirs(saveDirFigs)

    #\\\ COMPUTE STATISTICS:
    # The first thing to do is to transform those into a matrix with all the
    # realizations, so create the variables to save that.
    meanLossTrainPerGraph = {}
    meanCostTrainPerGraph = {}
    meanLossValidPerGraph = {}
    meanCostValidPerGraph = {}
    meanLossTrain = {}
    meanCostTrain = {}
    meanLossValid = {}
    meanCostValid = {}
    stdDevLossTrain = {}
    stdDevCostTrain = {}
    stdDevLossValid = {}
    stdDevCostValid = {}
    # Initialize the variables
    for thisModel in modelList:
        meanLossTrainPerGraph[thisModel] = [None] * nGraphRealizations
        meanCostTrainPerGraph[thisModel] = [None] * nGraphRealizations
        meanLossValidPerGraph[thisModel] = [None] * nGraphRealizations
        meanCostValidPerGraph[thisModel] = [None] * nGraphRealizations
        if nGraphRealizations > 1:
            for G in range(nGraphRealizations):
                # Transform into np.array
                lossTrain[thisModel][G] = np.array(lossTrain[thisModel][G])
                costTrain[thisModel][G] = np.array(costTrain[thisModel][G])
                lossValid[thisModel][G] = np.array(lossValid[thisModel][G])
                costValid[thisModel][G] = np.array(costValid[thisModel][G])
                # So, finally, for each model and each graph, we have a np.array of
                # shape:  nDataRealizations x number_of_training_steps
                # And we have to average these to get the mean across all data
                # realizations for each graph
                meanLossTrainPerGraph[thisModel][G] = \
                                    np.mean(lossTrain[thisModel][G], axis = 0)
                meanCostTrainPerGraph[thisModel][G] = \
                                    np.mean(costTrain[thisModel][G], axis = 0)
                meanLossValidPerGraph[thisModel][G] = \
                                    np.mean(lossValid[thisModel][G], axis = 0)
                meanCostValidPerGraph[thisModel][G] = \
                                    np.mean(costValid[thisModel][G], axis = 0)
        else:
            meanLossTrainPerGraph[thisModel] = lossTrain[thisModel][0]
            meanCostTrainPerGraph[thisModel] = costTrain[thisModel][0]
            meanLossValidPerGraph[thisModel] = lossValid[thisModel][0]
            meanCostValidPerGraph[thisModel] = costValid[thisModel][0]
        # And then convert this into np.array for all graphs
        meanLossTrainPerGraph[thisModel] = \
                                    np.array(meanLossTrainPerGraph[thisModel])
        meanCostTrainPerGraph[thisModel] = \
                                    np.array(meanCostTrainPerGraph[thisModel])
        meanLossValidPerGraph[thisModel] = \
                                    np.array(meanLossValidPerGraph[thisModel])
        meanCostValidPerGraph[thisModel] = \
                                    np.array(meanCostValidPerGraph[thisModel])
        # And compute the statistics
        meanLossTrain[thisModel] = \
                            np.mean(meanLossTrainPerGraph[thisModel], axis = 0)
        meanCostTrain[thisModel] = \
                            np.mean(meanCostTrainPerGraph[thisModel], axis = 0)
        meanLossValid[thisModel] = \
                            np.mean(meanLossValidPerGraph[thisModel], axis = 0)
        meanCostValid[thisModel] = \
                            np.mean(meanCostValidPerGraph[thisModel], axis = 0)
        stdDevLossTrain[thisModel] = \
                            np.std(meanLossTrainPerGraph[thisModel], axis = 0)
        stdDevCostTrain[thisModel] = \
                            np.std(meanCostTrainPerGraph[thisModel], axis = 0)
        stdDevLossValid[thisModel] = \
                            np.std(meanLossValidPerGraph[thisModel], axis = 0)
        stdDevCostValid[thisModel] = \
                            np.std(meanCostValidPerGraph[thisModel], axis = 0)

    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
    varsPickle = {}
    varsPickle['nEpochs'] = nEpochs
    varsPickle['nBatches'] = nBatches
    varsPickle['meanLossTrain'] = meanLossTrain
    varsPickle['stdDevLossTrain'] = stdDevLossTrain
    varsPickle['meanCostTrain'] = meanCostTrain
    varsPickle['stdDevCostTrain'] = stdDevCostTrain
    varsPickle['meanLossValid'] = meanLossValid
    varsPickle['stdDevLossValid'] = stdDevLossValid
    varsPickle['meanCostValid'] = meanCostValid
    varsPickle['stdDevCostValid'] = stdDevCostValid
    with open(os.path.join(saveDirFigs,'figVars.pkl'), 'wb') as figVarsFile:
        pickle.dump(varsPickle, figVarsFile)

    ########
    # PLOT #
    ########

    # Compute the x-axis
    xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
    xValid = np.arange(0, nEpochs * nBatches, \
                          validationInterval*xAxisMultiplierValid)

    # If we do not want to plot all the elements (to avoid overcrowded plots)
    # we need to recompute the x axis and take those elements corresponding
    # to the training steps we want to plot
    if xAxisMultiplierTrain > 1:
        # Actual selected samples
        selectSamplesTrain = xTrain
        # Go and fetch tem
        for thisModel in modelList:
            meanLossTrain[thisModel] = meanLossTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevLossTrain[thisModel] = stdDevLossTrain[thisModel]\
                                                        [selectSamplesTrain]
            meanCostTrain[thisModel] = meanCostTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevCostTrain[thisModel] = stdDevCostTrain[thisModel]\
                                                        [selectSamplesTrain]
    # And same for the validation, if necessary.
    if xAxisMultiplierValid > 1:
        selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
                                       xAxisMultiplierValid)
        for thisModel in modelList:
            meanLossValid[thisModel] = meanLossValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevLossValid[thisModel] = stdDevLossValid[thisModel]\
                                                        [selectSamplesValid]
            meanCostValid[thisModel] = meanCostValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevCostValid[thisModel] = stdDevCostValid[thisModel]\
                                                        [selectSamplesValid]

    #\\\ LOSS (Training and validation) for EACH MODEL
    for key in meanLossTrain.keys():
        lossFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.errorbar(xValid, meanLossValid[key], yerr = stdDevLossValid[key],
                     color = '#95001A', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        lossFig.savefig(os.path.join(saveDirFigs,'loss%s.pdf' % key),
                        bbox_inches = 'tight')

    #\\\ RMSE (Training and validation) for EACH MODEL
    for key in meanCostTrain.keys():
        costFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
        plt.errorbar(xTrain, meanCostTrain[key], yerr = stdDevCostTrain[key],
                     color = '#01256E', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.errorbar(xValid, meanCostValid[key], yerr = stdDevCostValid[key],
                     color = '#95001A', linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
        plt.ylabel(r'Error rate')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        costFig.savefig(os.path.join(saveDirFigs,'cost%s.pdf' % key),
                        bbox_inches = 'tight')

    # LOSS (training) for ALL MODELS
    allLossTrain = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanLossTrain.keys():
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'Loss')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanLossTrain.keys()))
    allLossTrain.savefig(os.path.join(saveDirFigs,'allLossTrain.pdf'),
                    bbox_inches = 'tight')

    # RMSE (validation) for ALL MODELS
    allCostValidFig = plt.figure(figsize=(1.61*figSize, 1*figSize))
    for key in meanCostValid.keys():
        plt.errorbar(xValid, meanCostValid[key], yerr = stdDevCostValid[key],
                     linewidth = lineWidth,
                     marker = markerShape, markersize = markerSize)
    plt.ylabel(r'Error rate')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanCostValid.keys()))
    allCostValidFig.savefig(os.path.join(saveDirFigs,'allCostValid.pdf'),
                    bbox_inches = 'tight')

# Finish measuring time
endRunTime = datetime.datetime.now()

totalRunTime = abs(endRunTime - startRunTime)
totalRunTimeH = int(divmod(totalRunTime.total_seconds(), 3600)[0])
totalRunTimeM, totalRunTimeS = \
               divmod(totalRunTime.total_seconds() - totalRunTimeH * 3600., 60)
totalRunTimeM = int(totalRunTimeM)

if doPrint:
    print(" ")
    print("Simulation started: %s" %startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Simulation ended:   %s" % endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    print("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                         totalRunTimeM,
                                         totalRunTimeS))
    
# And save this info into the .txt file as well
with open(varsFile, 'a+') as file:
    file.write("Simulation started: %s\n" % 
                                     startRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Simulation ended:   %s\n" % 
                                       endRunTime.strftime("%Y/%m/%d %H:%M:%S"))
    file.write("Total time: %dh %dm %.2fs" % (totalRunTimeH,
                                              totalRunTimeM,
                                              totalRunTimeS))
