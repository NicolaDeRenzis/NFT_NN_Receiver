#import matplotlib.pyplot as plt
import sys, os, math
import scipy.io
import h5py # need to load .mat files
import numpy as np
import tensorflow as tf
# sys.path.append("/work1/rajo/setup_nftTransmission/neuralNetwork")
sys.path.append("/zhome/df/1/113755/Code/NFT_NN_receiver/code/auxCode")
import NeuralNetwork # imports the functions from the path above
import traceLoader

import argparse
parser = argparse.ArgumentParser(description='Submit Job Array to cluster')
parser.add_argument('-n', type=int, help='LSB Job Index', required=True)
parser.add_argument('-m', type=str, help='mat file to load', required=True)
parser.add_argument('-p', type=str, help='path where to save results', required=True)
parser.add_argument('-o', type=str, help='operation mode', required=True)
args = vars(parser.parse_args())
#print('LSB Job Index:', args['n'])

idx = int(args['n'])
matlabFile = str(args['m'])
mode = str(args['o'])
pathToSave = str(args['p'])

#rootPath = '/zhome/df/1/113755/Code/NFT_NN_receiver/code/NN_train_2'



############ CHOSE MODE TO OPERATE

batchSize = 1000;
if mode == 'regression':
	activationFcn = 'sigmoid';
	nHiddneUnits = 32;
	if idx>=15:
		batchSize = 3500;
	else:
		batchSize = 5000;

elif mode == 'classification':
	activationFcn = 'tanh';
	if idx>=12:
		nHiddneUnits = 64;
		batchSize = 2500;
	elif idx<=8:
		nHiddneUnits = 256;
		batchSize = 2500;
	else:
		nHiddneUnits = 32;
		batchSize = 2500;

#mode = 'classification'; # or 'regression'


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

#idx = int(60); # idx to save the NN parameters

############

case = 'single'
print('OSNR: ', idx)
print('hidden Units: ', nHiddneUnits)
print('mode: ', mode)
print('batchSize: ', batchSize);

# Load MAT file
output = h5py.File(matlabFile)
print(matlabFile)

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
params.nHiddenUnits     = nHiddneUnits # PhaseNoise in B2B
params.idx              = idx
params.sweep_idx        = 1
params.learning_rate    = 0.01;
params.seed             = 1
params.activation       = activationFcn;
params.batch_size       = batchSize; #params.N_train; # 2500
params.mode             = mode

attributes = [(attr,getattr(params, attr)) for attr in dir(params) if not attr.startswith('__')]
print(attributes)

# Param path
paths = NeuralNetwork.defaultPaths()
paths.checkpointDir     = case
paths.saveRoot          = os.path.join(pathToSave, 'trainedNN', mode)
paths.checkpointRoot    = os.path.join(paths.saveRoot, 'tflowCheckpoints')
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

#X = X + 0.01*np.random.normal(loc=0.0, scale=1.0, size=X.shape);

print(traceParams.samples)
print(X.shape)


if mode == 'classification':
    maxPoss = traceParams.nModes*np.power(traceParams.M,(traceParams.nEigs));
    Y_ = tf.cast( tf.one_hot(traceParams.targets ,maxPoss), tf.uint8)
    sess = tf.Session()
    Y = sess.run(Y_);
    Y = Y[0,:,:];
    
elif mode == 'regression':
    Y = traceParams.targets;


NeuralNetwork.train(X,Y,inputNorm,params,paths)


