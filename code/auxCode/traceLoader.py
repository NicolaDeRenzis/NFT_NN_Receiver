import sys, os, math
import scipy.io
import h5py # need to load .mat files
import numpy as np

class TraceLoader:
    paramsToFilter = dict();
    
    def traceFilter(self, filterParams):
        for i in filterParams:
            self.paramsToFilter.update({i : filterParams[i]});
            
    def loadTrace(self, matfilesPath, toLoad = None):
        matFiles = self.listFiles(self, matfilesPath);
        correctTracesName = [];
        for i in matFiles:
            currentParams = dict();
            current = i.replace('.mat', '')
            splitted = current.split('_')
            for k in splitted:
                if k.split('=')[0] == k:
                    splitted.remove(k)
                else:
                    currentParams.update({k.split('=')[0] : np.float32(k.split('=')[1])} )
                    
            if currentParams.items() >= self.paramsToFilter.items():
                correctTracesName.append(i)
                
        traceToLoad = toLoad if toLoad is not None else 0
        if len(correctTracesName) > 0:
            fname = os.path.join(matfilesPath, correctTracesName[traceToLoad])
            self.loadedTrace = correctTracesName[traceToLoad]
            return h5py.File(fname)
        else:
            self.loadedTrace = [];
            return -1
        
    def listFiles(self, matfilesPath):
        matfiles = [f for f in os.listdir(matfilesPath) if os.path.isfile(os.path.join(matfilesPath, f))]
        matfiles.sort()
        return matfiles
    
     
    