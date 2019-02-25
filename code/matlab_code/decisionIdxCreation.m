ccc

% signal generated path
load('/work3/nidre/NN_receiver/generationSignal/10-43-28__rngSeed=1.mat');

bits = logical(traceAndSetupParameters.txResults.txDiscrete.inputBitSequencesDiscrete.get);

decisionIdx = bi2de(bits)+1;

save('~/Code/NFT_NN_receiver/code/matlab_code/decisionMatrix.mat', 'decisionIdx');