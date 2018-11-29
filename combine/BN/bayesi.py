import scipy.io as scio
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mat4py
from scipy.sparse.csgraph import minimum_spanning_tree


data_train = mat4py.loadmat('traindata.mat')
data_train = data_train['traindata']
data_train_label = mat4py.loadmat('trainlabelbe.mat')
data_train_label = data_train_label['trainlabel']
data_test = mat4py.loadmat('datatest.mat')
data_test = data_test['datatest']
data_test_label = mat4py.loadmat('testlabelbe.mat')
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
#train_data_pad = np.concatenate( ( np.ones((data_train.shape[0], 1)), data_train ), axis = 1 )
#test_data_pad = np.concatenate( ( np.ones((data_test.shape[0], 1)), data_test ), axis = 1 )
train_data_pad = data_train
test_data_pad = data_test
print(np.shape(train_data_pad))
print(np.shape(test_data_pad))
print(np.shape(data_train_label.T[0]))
def corr(x,y):
    if x.shape[0] == 0:
        rowvar = 1
    else:
        rowvar = 0
    return np.corrcoef(np.asarray(x),np.asarray(y),rowvar = rowvar)[0,1]

def mutual_information(x,y):
    return -(1.0/2)*np.log(1-corr(x,y)**2)

def mutual_info_all(M):
    f_num = M.shape[1] #feature number
    mi_ary = np.zeros( (f_num, f_num) )
    for i in range(f_num):
        for j in range(i+1,f_num):
            mi_ary[i,j] = mutual_information(M[:,i],M[:,j])
    return mi_ary

def chowliu(features):
    M = mutual_info_all(features)
    adjacency_matrix = minimum_spanning_tree(-M)
    return adjacency_matrix

def get_edge_list(mat):
    f_num = mat.shape[0]
    edges = []
    for k in range(f_num):
        lst = np.nonzero(mat[k,:])[1]
        # k is parent, j is child
        new_edges = [(k,j) for j in lst]        
        edges.extend(new_edges)
    #for i in range(len(edges)):
    #    if edges[i][0] > edges[i][1]:
    #       edges[i]=edges[i][::-1]
    return edges

def get_label_subsets(data_train_label):
    label_set = np.unique(data_train_label) #get 5 label numbers
    label_sample_map = {} #a label to sample index map
    for i in label_set:
        j = np.argwhere(data_train_label == i).T
        label_sample_map[i] = j
    return label_sample_map

def get_edges_for_each_class(train_data_pad, data_train_label):
    label_set = np.unique(data_train_label) #get 5 label numbers
    print(label_set)
    label_sample_map = get_label_subsets(data_train_label.T[0]) 
    print(label_sample_map)
    class_edges = {}
    for i in label_set:
        features = train_data_pad[label_sample_map[i],:]
        
        print(features.shape[0],features.shape[1],features.shape[2])
        features = np.reshape(features, (features.shape[1],features.shape[2]))
        adjacency = chowliu(features)
        class_edges[i] = get_edge_list(adjacency)
    return class_edges

class_edges = get_edges_for_each_class(train_data_pad, data_train_label) 

def compute_theta_r( x ):
    return np.mean(x[:,0])

def compute_theta_j_k_0( j, k, x, theta_j_1 ):
    return np.mean(x[:,j]-theta_j_1*x[:,k])

def compute_theta_j_k_1( j, k, x, theta_j_0 ):
    return np.sum(x[:,k]*(x[:,j]-theta_j_0))/np.sum(x[:,k]**2)

def get_thetas_for_each_class(train_data_pad, data_train_label, class_edges):
    label_sample_map = get_label_subsets(data_train_label.T[0])
    label_set = np.unique(data_train_label)
    class_thetas = {}
    f_num = train_data_pad.shape[1]
    for lab in label_set:
        print( 'processing class label {}'.format(lab) )
        c_samples = train_data_pad[label_sample_map[lab],:]
        c_samples = np.reshape(c_samples, (c_samples.shape[1],c_samples.shape[2]))
        theta_r = compute_theta_r(c_samples)
        c_edge_list= class_edges[lab]
        thetas = np.zeros((f_num,2)) #the first column shoud be j_0, and the second column should be j_1
        #the first row (thetas[0,:]) is for theta_r

        for (k,j) in c_edge_list: 
            theta_j_1 = 0
            #should do coordinate ascent using the function
            #compute_theta_j_k_0 and compute_theta_j_k_1 here
            for z in range(40):
                theta_j_0 = compute_theta_j_k_0( j, k, c_samples, theta_j_1)
                theta_j_1 = compute_theta_j_k_1( j, k, c_samples, theta_j_0)
            #set the optimal theta_j_0 and theta_j_1 for this the edge (k, j)
            thetas[j, 0] = theta_j_0
            thetas[j, 1] = theta_j_1
        thetas[0,0] = theta_r   
        # root has no parents
        thetas[0,1] = np.nan
        class_thetas[lab] = thetas
    return class_thetas
class_thetas = get_thetas_for_each_class(train_data_pad,data_train_label,class_edges)

def compute_lp_j_k( j, k, x, theta_j_0, theta_j_1 ):
    return -np.log( 2 * np.pi )-0.5*( x[j] - theta_j_0 - theta_j_1 * x[k] )**2

def compute_lp_r( x, theta_r ):
    return -np.log( 2 * np.pi )-0.5*( x[0] - theta_r[0] )**2

def compute_lp_x_given_Theta( x, thetas, edges):
    lp = compute_lp_r(x,thetas[0,:])
    for (k,j) in edges:
        # k is parent, j is child
        lp += compute_lp_j_k( j, k, x, thetas[j,0], thetas[j,1] )
    return lp

def get_log_p_h(data_train_label):
    label_set = np.unique(data_train_label)
    log_p_h = np.zeros(2)
    for i in label_set:
        count = len(np.nonzero(data_train_label==i)[0])
        log_p_h[i] = np.log( float(count) / float(len(data_train_label) ) )
    return log_p_h

log_p_h = get_log_p_h(data_train_label)  

def logsumexp(vec):
    m = np.max(vec,axis=0)   
    return np.log(np.sum(np.exp(vec-m),axis=0))+m

def p_h_given_x_theta( x, class_edges, class_thetas, log_p_h ):
    
    C = len(class_thetas)
    lognumerator = np.zeros(C)
    
    # implement Bayes rule here
    # compute log-numerators first and then normalize using logsumexp
    # there are more compact ways to do the normalization
    # feel free to rearrange the code, as long as you return correct
    # probabilities
    for i in range(C): 
        edges = class_edges[i]
        thetas = class_thetas[i]        
        lognumerator[i] = compute_lp_x_given_Theta(x, thetas, edges) + log_p_h[i]
    
    # use logsumexp to compute denominator
    logdenominator = logsumexp(lognumerator)
        
    numerator = np.exp(lognumerator)
    denominator = np.exp(logdenominator)
    
    probs = numerator/denominator
    
    assert(np.all(probs >= 0))
    assert(np.abs(np.sum(probs)-1.0)<1e-5)
    return probs

def evaluate_predictions(test_data_pad,data_test_label,class_edges,class_thetas,log_p_h):
    label_set = np.unique(data_train_label)
    pred_lab = np.zeros((test_data_pad.shape[0], 2)) 
    #the first column shoud be the predicted label, and the second column should be the probability of that label.
    test_num = test_data_pad.shape[0]
    for i in range(test_num):
        x = test_data_pad[i,:]    
        res =  p_h_given_x_theta(x, class_edges,class_thetas, log_p_h)     
        # predicted label
        pred_lab[i, 0] = np.argmax(res)
        # probability of that label
        pred_lab[i, 1] = res[int(pred_lab[i, 0])]
    return pred_lab 

pred_lab = evaluate_predictions(test_data_pad,data_test_label,class_edges,class_thetas,log_p_h)
#print("Prediction Accuracy: {}".format(np.mean(pred_lab[:,0]==data_test_label)))
print(pred_lab[:,0])
print(data_test_label)

