import matplotlib.pyplot as plt
import sys, os, math
import scipy.io
import h5py # need to load .mat files
import numpy as np
import tensorflow as tf
# sys.path.append("/work1/rajo/setup_nftTransmission/neuralNetwork")
sys.path.append("C:\\Users\\nidre\\rasmus\\NFT_NN_Receiver\\code\\auxCode")
import NeuralNetwork # imports the functions from the path above
import traceLoader

import argparse
parser = argparse.ArgumentParser(description='Submit Job Array to cluster')
parser.add_argument('-n', type=int, help='LSB Job Index', required=True)
#args = vars(parser.parse_args())
#print('LSB Job Index:', args['n'])
#idx = int(args['n'])


rootPath = 'C:\\Users\\nidre\\rasmus\\NFT_NN_Receiver\\code\\NN_train_1'

nSimulations = 220

############ CHOSE MODE TO OPERATE

mode = 'regression'; # or 'regression'
# 1 eig:
#   1: regression, 32 hidden unit, sigmoid, lin, mse, no OSNR
#   2: regression, 32 hidden unit, sigmoid, lin, mse, no OSNR new file
#   3: regression, 32 hidden unit, sigmoid, lin, mse, OSNR = 20
#   4: regression, 32 hidden unit, sigmoid, lin, mse, OSNR = 10

# 2 eig:
#   20: regression, 64 hidden unit, sigmoid, lin, mse, no OSNR
#   30: regression, 32 hidden unit, sigmoid, lin, mse, OSNR = 20 batch 3500
#   40: regression, 32 hidden unit, sigmoid, lin, mse, OSNR = 10 batch 5000

# 1 eig:
#   5: classification, 64 hidden unit, tanh, softmax, crossEntr, OSNR = 20
#   6: classification, 64 hidden unit, tanh, softmax, crossEntr, OSNR = 10

# 2 eig:
#   50: classification, 64 hidden unit, tanh, softmax, crossEntr, OSNR = 20
#   60: classification, 32 hidden unit, tanh, softmax, crossEntr, OSNR = 10

idx = int(100); # idx to save the NN parameters

############

# idx and sweep
python_idx = idx-1
python_idx = int(python_idx%nSimulations)
case = 'single'
print('python_idx: ', python_idx)

# Load MAT file
#matfilesPath = 'traces';
#matfilesPath = 'traces_PNL';

if mode == 'classification':
    matfilesPath = 'traces_PN_OSNR'; # after the training for PhaseNoise
elif mode == 'regression': # to train for phaseNoise
    matfilesPath = 'traces_PN';
    
#matfilesPath = 'traces_PNL';    
filterParams = {
        'nEigenvalues'    : 2,
#        'bbfilterEnabled' : 0,
        'nPoints'         : 64,
#        'NNReceiverPN'    : 1,
#        'nNFDMSymbols'    : 5e5,
#        'linewidth'       : 1e4,
        'OSNR'            : 10} # 1e500
#}

traceLoaderObj = traceLoader.TraceLoader;
traceLoaderObj.traceFilter(traceLoaderObj, filterParams);
output = traceLoaderObj.loadTrace(traceLoaderObj, matfilesPath);
print(traceLoaderObj.loadedTrace)

X = np.transpose(np.array(output['Y'])); # Y[:,0] is real signal, Y[:,1] is imaginary
class traceParams:
    exist = True;
    
traceParams.samples = np.int(np.array(output['traceAndSetupParameters/INFT/nPoints']));
traceParams.nModes = np.int(np.array(output['traceAndSetupParameters/tx/nModes']));
traceParams.nEigs = np.int(np.array(output['traceAndSetupParameters/txDiscrete/nEigenvalues']));
traceParams.M = np.int(np.array(output['traceAndSetupParameters/txDiscrete/M']));
traceParams.nSymbols = np.int(np.array(output['traceAndSetupParameters/tx/nNFDMSymbols']));
traceParams.targets = np.array(output['targets']);

print(X.shape)

# Param
params = NeuralNetwork.defaultParams()
params.trainingData     = traceParams.nSymbols
params.N_train          = int( 0.9*params.trainingData )
params.N_test           = int( 0.1*params.trainingData )
params.beta             = 0
#params.nHiddenUnits     = 32 # 64 good for 1 eig, 32 for 2 eig. No Phase noise in B2B
params.nHiddenUnits     = 32 # PhaseNoise in B2B
params.idx              = python_idx+1
params.sweep_idx        = 1
params.learning_rate    = 0.01;
params.seed             = 1
params.activation       = 'tanh';
params.batch_size       = 2500; #params.N_train; # 2500
params.mode             = mode

attributes = [(attr,getattr(params, attr)) for attr in dir(params) if not attr.startswith('__')]
print(attributes)

# Param path
paths = NeuralNetwork.defaultPaths()
paths.checkpointRoot    = os.path.join(rootPath, 'tflowCheckpoints')
paths.checkpointDir     = case
paths.saveRoot          = os.path.join(rootPath, 'trainedNN', mode)
paths.saveDir           = case
paths.savePrefix        = 'NN'



inputNorm = np.max(X[:,:])
# the next line concatenates real and imag part of the signal, grouping the sampels by NFDM symbols
X = np.concatenate( (X[:,0].reshape((traceParams.nSymbols,traceParams.samples)),X[:,1].reshape((traceParams.nSymbols,traceParams.samples))),axis=1)/inputNorm
X_tmp = X;

## add more symbols for phase noise 
S = 1; # symbols to input
for i in range(S-1):
    k=i+1+1;
    X_aux = np.roll(X_tmp, k-1, axis=0);
    X = np.concatenate((X_aux, X), axis=1);

#X = X + 0.01*np.random.normal(loc=0.0, scale=1.0, size=X.shape)

plt.plot(X[0,:])
plt.plot(X[3,:])
plt.plot(X[5,:])
plt.plot(X[13,:])
plt.plot(X[14,:])
plt.plot(X[17,:])
plt.show();
#plt.plot(X[0,:])
#plt.plot(X[55,:])
#plt.plot(X[83,:])
#plt.plot(X[99,:])
#plt.plot(X[112,:])
#plt.show();

print(traceParams.samples)
print(X.shape)

#tmp = scipy.io.loadmat('decisionMatrix.mat');
#Y = h5py.File('nEigenvalues=2_rngSeed=1_nPoints=64-decisionMatrix.mat')
#Y = np.array(Y['decisionIdx'])-1;
#maxPoss = traceParams.nModes*np.power(traceParams.M,(traceParams.nEigs));

#Y = h5py.File('phaseNoise_1_regres.mat')
#traceParams.targets = np.array(Y['phaseNoiseScrumbled']);


if mode == 'classification':
    maxPoss = traceParams.nModes*np.power(traceParams.M,(traceParams.nEigs));
    Y_ = tf.cast( tf.one_hot(traceParams.targets ,maxPoss), tf.uint8)
    sess = tf.Session()
    Y = sess.run(Y_);
    Y = Y[0,:,:];
    
elif mode == 'regression':
    Y = traceParams.targets;

#Y = h5py.File('phaseNoise_1.mat')
#Y = np.transpose(np.array(Y['phaseNoise_hot'])-1);
#maxPoss = 90;
#
#
#
#Y_ = tf.cast( tf.one_hot(Y,maxPoss), tf.uint8)
#sess = tf.Session()
#Y = sess.run(Y_);
#Y = Y[0,:,:];

#print( Y[ [0,3,5,13,14,17], : ] )
#print( Y[ [0,55,83,99,112], : ] )

#Y = h5py.File('phaseNoise_1_regres_pn.mat')
#Y = np.array(Y['phaseNoiseScrumbled']);

#plt.figure();
#plt.plot(X[3,:])
#plt.show();


NeuralNetwork.train(X,Y,inputNorm,params,paths)




