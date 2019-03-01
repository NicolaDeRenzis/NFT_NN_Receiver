import matplotlib.pyplot as plt
import sys, os, math
import scipy.io
import h5py # need to load .mat files
import numpy as np
import tensorflow as tf
# sys.path.append("/work1/rajo/setup_nftTransmission/neuralNetwork")
sys.path.append("C:\\Users\\nidre\\rasmus\\NFT_NN_Receiver\\code\\neuralNetwork")
import NeuralNetwork # imports the functions from the path above

import argparse
parser = argparse.ArgumentParser(description='Submit Job Array to cluster')
parser.add_argument('-n', type=int, help='LSB Job Index', required=True)
#args = vars(parser.parse_args())
#print('LSB Job Index:', args['n'])
#idx = int(args['n'])
idx = int(1);

rootPath = 'C:\\Users\\nidre\\rasmus\\NFT_NN_Receiver\\code\\NN_train_1'

nSimulations = 120

# idx and sweep
python_idx = idx-1
python_idx = int(python_idx%nSimulations)
case = 'single'
print('python_idx: ', python_idx)

# Load MAT file
matfilesPath = 'traces';
class filterParams:
    exist = True;
#    def getSingleTrace(self, listOfTraces):
#        for i in length self.fields 
        
    
filterParams.nPoints = 64;
matfiles = NeuralNetwork.listFiles(matfilesPath)

fname = os.path.join(matfilesPath,matfiles[python_idx])
print(fname)
output = h5py.File(fname);
#output = scipy.io.loadmat(fname)

X = np.transpose(np.array(output['Y'])); # Y[:,0] is real signal, Y[:,1] is imaginary
class traceParams:
    exist = True;
    
traceParams.samples = np.int(np.array(output['traceAndSetupParameters/INFT/nPoints']));
traceParams.nModes = np.int(np.array(output['traceAndSetupParameters/tx/nModes']));
traceParams.nEigs = np.int(np.array(output['traceAndSetupParameters/txDiscrete/nEigenvalues']));
traceParams.M = np.int(np.array(output['traceAndSetupParameters/txDiscrete/M']));
traceParams.nSymbols = np.int(np.array(output['traceAndSetupParameters/tx/nNFDMSymbols']));

print(X.shape)

default = {'beta': 0,
           'nHiddenUnits': traceParams.nModes*traceParams.samples*2*2,
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
params.learning_rate    = 0.01
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

print(traceParams.samples)
print(X.shape)

#tmp = scipy.io.loadmat('decisionMatrix.mat');
Y = h5py.File('09-39-27__OSNR=2e1_rngSeed=1-decisionMatrix.mat')
Y = np.array(Y['decisionIdx']);

maxPoss = traceParams.nModes*traceParams.nEigs*traceParams.M;
Y_ = tf.cast( tf.one_hot(Y,maxPoss), tf.uint8)
sess = tf.Session()
Y = sess.run(Y_);
Y = Y[0,:,:];
#plt.figure();
#plt.plot(X[3,:])
#plt.show();


NeuralNetwork.train(X,Y,inputNorm,params,paths)




