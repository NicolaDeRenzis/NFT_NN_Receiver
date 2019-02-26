ccc

%% load trace
traceFolder = '/work3/nidre/NN_receiver/generationSignal';
traceFilter = struct(); % Set it to load a specific trace (if multiple matches, loads the first only
traceFilter.OSNR = 20;
traceFilter.rngSeed = 1;
fileNames = traceFinder(traceFolder, traceFilter);
param.txTrLoad.loadFileName = [traceFolder filesep fileNames(1).name];

load(param.txTrLoad.loadFileName);

%% extract info
bits = logical(traceAndSetupParameters.txResults.txDiscrete.inputBitSequencesDiscrete.get);

decisionIdx = bi2de(bits)+1;

% save('~/Code/NFT_NN_receiver/code/matlab_code/decisionMatrix.mat', 'decisionIdx');
save([strrep(param.txTrLoad.loadFileName,'.mat',''), '_decisionMatrix.mat'], 'decisionIdx');