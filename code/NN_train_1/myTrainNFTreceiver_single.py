import sys, os, math
import scipy.io
import numpy as np
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

default = {'beta': 0,
           'nHiddenUnits': 128,
           'trainingData': 100000,
           'samples': 16}


# idx and sweep

python_idx = idx-1
python_idx = int(python_idx%nSimulations)
case = 'single'
print('python_idx: ', python_idx)

samples = default['samples']
matfilesPath = '../trace_201711160210_coherent_'+str(samples)

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

# Load MAT file
matfiles = NeuralNetwork.listFiles(matfilesPath)
fname = os.path.join(matfilesPath,matfiles[python_idx])
print(fname)
output = scipy.io.loadmat(fname)

X = output['Y']

print(X.shape)

inputNorm = np.max(X[:,:])
X = np.concatenate( (X[:,0].reshape((100000,samples)),X[:,1].reshape((100000,samples))),axis=1)/inputNorm

print(samples)
print(X.shape)

Y = scipy.io.loadmat('decisionMatrix.mat')
Y = Y['decisionMatrix']

NeuralNetwork.train(X,Y,inputNorm,params,paths)




