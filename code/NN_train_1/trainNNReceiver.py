import sys, os, math
import scipy.io
import numpy as np
# sys.path.append("/work1/rajo/setup_nftTransmission/neuralNetwork")
sys.path.append("C:\\Users\\nidre\\rasmus\\NFT_NN_Receiver\\code\\neuralNetwork")
import NeuralNetwork
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
import argparse
parser = argparse.ArgumentParser(description='Submit Job Array to cluster')
parser.add_argument('-n', type=int, help='LSB Job Index', required=True)
#args = vars(parser.parse_args())
#print('LSB Job Index:', args['n'])
#idx = int(args['n'])
idx = int(1);

rootPath = 'C:\\Users\\nidre\\rasmus\\NFT_NN_Receiver\\code\\NN_train_1'

cases = ['nHiddenUnits','trainingData','samples'] # 4*120 / 4*120 / 4*120
nSimulations = 120

default = {'beta':          0,
           'nHiddenUnits':  32,
           'trainingData':  100000,
           'samples':       32}
           
sweeper = {'beta':          [0,1e-7,1e-5,1e-3],
           'nHiddenUnits':  [4,8,16,32],
           'trainingData':  [10000, 25000, 75000, 100000],
           'samples':       [8, 16, 32, 128]}

# idx and sweep
python_idx, sweepIdx, case_idx, case = NeuralNetwork.arrangeIdx(idx,cases,sweeper,nSimulations)
print('python_idx, sweepIdx, case_idx: ',python_idx,sweepIdx, case_idx)

samples = (sweeper['samples'][sweepIdx] if case=='samples' else default['samples'])
matfilesPath = '../trace_201711160210_coherent_'+str(samples)

# Param
params = NeuralNetwork.defaultParams()
params.N_train          = int( 0.9*(sweeper['trainingData'][sweepIdx] if case=='trainingData' else default['trainingData']) )
params.N_test           = int( 0.1*(sweeper['trainingData'][sweepIdx] if case=='trainingData' else default['trainingData']) )
params.beta             = (sweeper['beta'][sweepIdx] if case=='beta' else default['beta'])
params.nHiddenUnits     = (sweeper['nHiddenUnits'][sweepIdx] if case=='nHiddenUnits' else default['nHiddenUnits'])
params.idx              = python_idx+1
params.sweep_idx        = sweepIdx+1
params.learning_rate    = 0.01
params.seed             = idx

attributes = [(attr,getattr(params, attr)) for attr in dir(params) if not attr.startswith('__')]
print(attributes,flush=True)

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

