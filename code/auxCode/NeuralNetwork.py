import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.io
import os, sys, math

class defaultParams:
    N_train=90000
    N_test=10000
    learning_rate=0.01
    batch_size=2500
    display_step = 10
    training_epochs=10000
    beta=0
    seed=1
    nHiddenUnits = 32
    idx=1
    sweep_idx=1
    nPCs=10

class defaultPaths:
    checkpointRoot      = ''
    checkpointDir       = ''
    saveRoot            = ''
    saveDir             = ''
    savePrefix          = ''
    
    def createFolders(self,idx1,idx2):
        self.makedirsIfNotExists( self.renderCheckpointPath(idx1, idx2) )
        self.makedirsIfNotExists( os.path.join(self.saveRoot,self.saveDir) )
            
    def makedirsIfNotExists(self,path):
        if not os.path.exists(path):
            os.makedirs(path)        
    
    def renderCheckpointPath(self, idx1, idx2):
        path = os.path.join(self.checkpointRoot,self.checkpointDir+'_{:04d}_{:04d}'.format(idx1,idx2))
        return path
        
    def renderSavePath(self,idx1,idx2):
        path = os.path.join(self.saveRoot, self.saveDir, self.savePrefix+'_{:04d}_{:04d}'.format(idx1,idx2))
        return path

def arrangeIdx(idx,cases,sweeper,nSimulations):
    
    python_idx = idx-1
    case_idx = 0
    runningSum = 0
    
    for c in cases:
        oldRunningSum = runningSum;
        runningSum += len(sweeper[c])
        if python_idx < runningSum*nSimulations:
            case = c
            case_idx = python_idx-oldRunningSum*nSimulations
            break
        
    sweepIdx = int( math.floor(case_idx/nSimulations) )
    python_idx = int(python_idx%nSimulations)
    
    return python_idx, sweepIdx, case_idx, case

def listFiles(matfilesPath):
    matfiles = [f for f in os.listdir(matfilesPath) if os.path.isfile(os.path.join(matfilesPath, f))]
    matfiles.sort()
    return matfiles

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#    layer_1 = tf.nn.sigmoid(layer_1)
    layer_1 = tf.nn.tanh(layer_1)
    # Output layer with linear activation
    out_layer = tf.nn.softmax(tf.matmul(layer_1, weights['out']) + biases['out'])
    return out_layer
    
def train(X,Y,inputNorm,params,paths):
    paths.createFolders(params.sweep_idx,params.idx)
    print(X.shape)    
    
    # In[25]:
    X_train = X[0:params.N_train,:]
    Y_train = Y[0:params.N_train,:]    
    X_test = X[params.N_train:params.N_train+params.N_test,:]
    Y_test = Y[params.N_train:params.N_train+params.N_test,:]    
    
    # In[26]:    
    print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
    print(X_test.shape, Y_test.shape)    
    
    # In[30]:
    n_input = X_train.shape[1]
    n_output = Y.shape[1]
    n_hidden_1 = params.nHiddenUnits
    
    # In[31]:
    tf.set_random_seed(params.seed)
    np.random.seed(params.seed)
    
    x = tf.placeholder(shape=[None, n_input], dtype=tf.float32)
    y = tf.placeholder(shape=[None, n_output], dtype=tf.float32)
    
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }
    
    pred = multilayer_perceptron(x, weights, biases)
#    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred), reduction_indices=[1])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
#    cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred);
    
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate).minimize(cost)    
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    
    
    # In[ ]:    
    init = tf.global_variables_initializer()
    session = tf.Session()
    saver = tf.train.Saver()        
    save_path = os.path.join(paths.renderCheckpointPath(params.sweep_idx,params.idx), 'best_test')
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    best_test = [];
    last_improvement = [];
    require_improvement = 20
    
    session.run(init)
    
    out_train_error = np.zeros(params.training_epochs)
    out_test_error = np.zeros(params.training_epochs)
    out_train_acc = np.zeros(params.training_epochs)
    out_test_acc = np.zeros(params.training_epochs)
    
    for epoch in range(params.training_epochs):
        avg_cost = 0.
        total_batch = int(params.N_train/params.batch_size)  
        shuffled_idx = np.arange(params.N_train)
        np.random.shuffle(shuffled_idx)
        for i in range(total_batch):
            batch_x_train = X_train[shuffled_idx[i:i+params.batch_size],:]
            batch_y_train = Y_train[shuffled_idx[i:i+params.batch_size],:]
            _, c = session.run([optimizer, cost], feed_dict={x: batch_x_train, y: batch_y_train})
            avg_cost += c / total_batch
        
        c_train, a_train = session.run([cost, accuracy], feed_dict={x: X_train, y: Y_train})
        c_test, a_test = session.run([cost, accuracy], feed_dict={x: X_test, y: Y_test})
        
        out_train_error[epoch] = c_train
        out_test_error[epoch] = c_test
        out_train_acc[epoch] = a_train
        out_test_acc[epoch] = a_test
        if epoch % params.display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "\ttrain=", "{:.9f}".format(c_train), "\ttest=", "{:.9f}".format(c_test) )
            print("Accuracy:", "\ttrain=", "{:.9f}".format(a_train), "\ttest=", "{:.9f}".format(a_test) )
            print("#####")
            
        if epoch == 0 or c_test < best_test:
            # Update the best-known validation accuracy.
            best_test = c_test
            # Set the iteration for the last improvement to current.
            last_improvement = epoch    
            # Save all variables of the TensorFlow graph to file.
            saver.save(sess=session, save_path=save_path)
            
        if epoch - last_improvement > require_improvement:
            print("Breaking due to no improvement: "+str(epoch))
            break
            
    saver.restore(sess=session, save_path=save_path)
    
    myVars = [x.eval(session=session) for x in tf.trainable_variables()] # session.run(x)
    for x in myVars:
        print(x.shape)
    
    scipy.io.savemat(paths.renderSavePath(params.sweep_idx,params.idx), {'w1':myVars[0],'w2':myVars[1],'b1':myVars[2],'b2':myVars[3],'inputNorm':inputNorm})
    
    plt.plot(range(1,epoch), out_test_acc[1:epoch])
    plt.show();
    plt.plot( range(1,epoch), out_test_error[1:epoch]);
    plt.show();
    plt.plot(range(1,epoch), out_train_acc[1:epoch])
    plt.show();
    plt.plot(range(1,epoch), out_train_error[1:epoch]);
    plt.show();