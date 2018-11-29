import scipy.io as scio
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mat4py


data_train = mat4py.loadmat('traindata.mat')
data_train = data_train['traindata']
data_train_label = mat4py.loadmat('trainlabel.mat')
data_train_label = data_train_label['trainlabel']
data_test = mat4py.loadmat('datatest.mat')
data_test = data_test['datatest']
data_test_label = mat4py.loadmat('testlabel.mat')
data_test_label = data_test_label['testlabel']
# obtain training and testing data
#data_path="traindata.mat"
#data = scio.loadmat(data_path)
#data_train=data.get('traindata')
#data_path="trainlabel.mat"
#data = scio.loadmat(data_path)
#data_train_label=data.get('trainlabel')
#data_path="datatest.mat"
#data = scio.loadmat(data_path)
#data_test=data.get('datatest')
#data_path="testlabel.mat"
#data = scio.loadmat(data_path)
#data_test_label=data.get('testlabel')
data_train = np.array(data_train)
data_train_label = np.array(data_train_label)
#data_train_label = data_train_label.T
data_test = np.array(data_test)
data_test_label = np.array(data_test_label)
#data_test_label = data_test_label.T
#print(np.shape(data_train))
#print(np.shape(data_train_label))
train_data_pad = np.concatenate( ( np.ones((data_train.shape[0], 1)), data_train ), axis = 1 )
test_data_pad = np.concatenate( ( np.ones((data_test.shape[0], 1)), data_test ), axis = 1 )

def loglikelihood(w, X, y, alpha): 
    #compute loglikelihood for current w, b, given the data X, y
    #w is a vector, b is a scalr, X is a n*p matrix and y is a vector.
    #Xtemp = np.ones(X.shape[0])
    #X = np.hstack((Xtemp,X))
    #print('u are coming')
    #print(np.shape(w))
    #print(np.shape(X))
    #print(np.shape(y))
    #print(np.shape(alpha))
    #print(alpha)
    usfr = np.dot(X, w)
    #print(np.shape(usfr))
    wnau = usfr.reshape((y.shape[0],1))
    #print(np.shape(wnau))
    tmp = 1. + np.exp( -np.multiply(y , wnau) )
    #print(y)
    #print(np.dot(X,w))
    #print(-y*np.dot(X,w))
    #print(np.shape(np.dot(X, w)))#471
    #print(np.shape(-y * (np.dot(X, w))))
    #print(np.shape(tmp))#471*1
    prob = 1./tmp
    #print(np.shape(prob))#471*1
    gtemp = y/(1+np.exp(y*wnau))# this is a n*1
    #print(np.shape(gtemp))#471*1
    X = X.T #X becomes a p*n matrix so the gradVal can be compute straight-forwardly.
    #print(np.shape(X))#5*471
    gradVal = np.dot(X, gtemp)
    #print(np.shape(gradVal))#5*1
    penalty = alpha/2.*np.sum(w[1:]*w[1:])
    #print(np.shape(penalty))#()
    gradPenalty = -alpha*w
    #print(np.shape(gradPenalty))#5
    gradPenalty[0] = 0;
    #print(np.shape(-np.sum( np.log( tmp ) ) - penalty))#()
    #print(np.shape(gradVal + gradPenalty.reshape((gradVal.shape[0],1))))#5*5
    return -np.sum( np.log( tmp ) ) - penalty, gradVal.reshape((gradVal.shape[0],)) + gradPenalty

def gradient_ascent(f,x,init_step,iterations):  
    f_val,grad = f(x)                           # compute function value and gradient 
    f_vals = [f_val]
    for it in range(iterations):                # iterate for a fixed number of iterations
        #print 'iteration %d' % it
        done = False                            # initial condition for done
        line_search_it = 0                      # how many times we tried to shrink the step
        step = init_step                        # reset step size to the initial size
        while not done and line_search_it<100:  # are we done yet?
            new_x = x + step*grad               # take a step along the gradient
            new_f_val,new_grad = f(new_x)       # evaluate function value and gradient
            if new_f_val<f_val:                 # did we go too far?
                step = step*0.95                # if so, shrink the step-size
                line_search_it += 1             # how many times did we shrank the step
            else:
                done = True                     # better than the last x, so we move on
        
        if not done:                            # did not find right step size
            print("Line Search failed.")
        else:
            f_val = new_f_val                   # ah, we are ok, accept the new x
            x = new_x
            grad = new_grad
            f_vals.append(f_val)
        plt.plot(f_vals)
    plt.xlabel('Iterations')
    plt.ylabel('Function value')
    return f_val, x

np.random.seed(12345)
w_init = np.random.randn( train_data_pad.shape[1] )*0.001
w_init[0] = 0

def optimizeFn( init_step, iterations, alpha, w):
    g = lambda xy0: loglikelihood(xy0, train_data_pad, data_train_label, alpha)
    #print(np.shape(g))
    #print(np.shape(w))
    f_val, update_w = gradient_ascent( g, w, init_step, iterations )
    return f_val, update_w

def prediction(w, data_test ):
    prob = 1./(1+np.exp(-np.dot(data_test,w)) );
    res = np.zeros(data_test.shape[0])
    res[prob>=0.5] = 1
    res[prob<0.5] = -1
    return res

f_val, update_w=optimizeFn( init_step = 1e-5, iterations=100, alpha=3000, w=w_init) 
#try different alphas [1000, 2000, 3000]
pred = prediction(update_w, test_data_pad)

print( 'accuracy on the test set {:.2f}%'.format( 100.*np.mean(pred==data_test_label)) )
print(pred.reshape((data_test_label.shape[0],1)))
print(data_test_label)
#wrong_idx = np.nonzero( data_test_label != pred )[0] #use this command to get the samples that are predicted wrong

#def computeProb(w, data_test ):
#    prob = 1./(1+np.exp(-np.dot(data_test,w)) )
#    return prob

#probs = computeProb(update_w, test_data_pad)
#wrong_idx_high = [i for i, v in enumerate(probs) if v > 0.9]
#print(wrong_idx_high)
#print(np.random.choice(wrong_idx_high,1))


