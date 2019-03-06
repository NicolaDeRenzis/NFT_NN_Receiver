import matplotlib.pyplot as plt
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
#args = vars(parser.parse_args())
#print('LSB Job Index:', args['n'])
#idx = int(args['n'])
idx = int(1);

rootPath = '/zhome/df/1/113755/Code/NFT_NN_receiver/code/NN_train_2'

nSimulations = 120

# idx and sweep
python_idx = idx-1
python_idx = int(python_idx%nSimulations)
case = 'single'
print('python_idx: ', python_idx)

# Load MAT file
matfilesPath = '/work2/nidre/equalizers/generation/toCluster/lossless_noiseless/DCh_spans_sweep';
filterParams = {
        'nEigenvalues'    : 1,
#        'bbfilterEnabled' : 1,
#        'nPoints'         : 64,
#        'OSNR'            : 10,
        'nSpans'          : 20,
        'inspectLength'   : 1e3}

traceLoaderObj = traceLoader.TraceLoader;
traceLoaderObj.traceFilter(traceLoaderObj, filterParams);
output = traceLoaderObj.loadTrace(traceLoaderObj, matfilesPath);
print(traceLoaderObj.loadedTrace)

class traceParams:
    exist = True;
### NEW CONFIG PARAMS     
#traceParams.samples = np.int(np.array(output['traceAndSetupParameters/INFT/nPoints']));
#traceParams.nModes = np.int(np.array(output['traceAndSetupParameters/tx/nModes']));
#traceParams.nEigs = np.int(np.array(output['traceAndSetupParameters/txDiscrete/nEigenvalues']));
#traceParams.M = np.int(np.array(output['traceAndSetupParameters/txDiscrete/M']));
#traceParams.nSymbols = np.int(np.array(output['traceAndSetupParameters/tx/nNFDMSymbols']));

## OLD CONFIG PARAMS
if traceLoaderObj.version == '-v6':
    traceParams.samples = np.int(np.array(output['traceAndSetupParameters']['INFT'][0][0]['nPoints']));
    traceParams.nModes = np.int(np.array(output['traceAndSetupParameters']['nModes']));
    traceParams.nEigs = np.int(np.array(output['traceAndSetupParameters']['nEigenvalues']));
    traceParams.M = np.int(np.array(output['traceAndSetupParameters']['M']));
    traceParams.nSymbols = np.int(np.array(output['traceAndSetupParameters']['nNFDMSymbols']));
    X = np.array(output['Y']); # Y[:,0] is real signal, Y[:,1] is imaginary
else:
    X = np.transpose(np.array(output['Y'])); # Y[:,0] is real signal, Y[:,1] is imaginary
print(X.shape)

default = {'beta': 0,
#           'nHiddenUnits': traceParams.nModes*traceParams.samples*2*2,
           'nHiddenUnits': 32, # 64 good for 1 eig
           'trainingData': traceParams.nSymbols,
           'samples': traceParams.samples}
# Param
params = NeuralNetwork.defaultParams()
params.N_train          = int( 0.9*default['trainingData'] )
params.N_test           = int( 0.1*default['trainingData'] )
params.beta             = default['beta']
params.nHiddenUnits     = default['nHiddenUnits']
params.idx              = python_idx+1
params.sweep_idx        = 1
params.learning_rate    = 0.01;
params.seed             = 1


attributes = [(attr,getattr(params, attr)) for attr in dir(params) if not attr.startswith('__')]
print(attributes)

# Param path
paths = NeuralNetwork.defaultPaths()
paths.checkpointRoot    = os.path.join(rootPath, 'tflowCheckpoints')
paths.checkpointDir     = case
paths.saveRoot          = os.path.join(rootPath, 'trainedNN')
paths.saveDir           = case
paths.savePrefix        = 'NN'



inputNorm = np.max(X[:,:])
# the next line concatenates real and imag part of the signal, grouping the sampels by NFDM symbols
X = np.concatenate( (X[:,0].reshape((traceParams.nSymbols,traceParams.samples)),X[:,1].reshape((traceParams.nSymbols,traceParams.samples))),axis=1)/inputNorm

X = X + 0.05*np.random.normal(loc=0.0, scale=1.0, size=X.shape)
plt.figure()
plt.plot(X[0,:])
plt.plot(X[3,:])
plt.plot(X[5,:])
plt.plot(X[13,:])
plt.plot(X[14,:])
plt.plot(X[17,:])

print(traceParams.samples)
print(X.shape)

#tmp = scipy.io.loadmat('decisionMatrix.mat');
# create decision matrix
Y = h5py.File(os.path.join(matfilesPath,'decisionMatrix.mat'))
Y = np.array(Y['decisionIdx'])-1;

maxPoss = traceParams.nModes*np.power(traceParams.M,(traceParams.nEigs));
Y_ = tf.cast( tf.one_hot(Y,maxPoss), tf.uint8)
sess = tf.Session()
Y = sess.run(Y_);
Y = Y[0,:,:];
print( Y[ [0,3,5,13,14,17], : ] )

#plt.figure();
#plt.plot(X[3,:])
#plt.show();


NeuralNetwork.train(X,Y,inputNorm,params,paths)




