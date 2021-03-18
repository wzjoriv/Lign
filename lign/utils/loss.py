import torch as th

from .clustering import similarity_matrix as sm

def similarity_matrix(x, p = 2): #pairwise distance

    n = x.size(0)
    d = x.size(1)

    y = x.unsqueeze(0).expand(n, n, d)
    x = x.unsqueeze(1).expand(n, n, d)
    
    dist = th.pow(x - y, p).sum(2)
    
    return dist

def vector_pairwise_diff(x):
    n = x.size(0)
    d = x.size(1)

    y = x.unsqueeze(0).expand(n, n, d)
    x = x.unsqueeze(1).expand(n, n, d)
    
    return x - y

def same_label(y): # return matric of nodes that have same label
    s = y.size(0)
    y_expand = y.unsqueeze(0).expand(s, s)
    Y = y_expand.eq(y_expand.t())
    return Y

def distance_loss(output, labels, Lambda = 0.01):

    ln = len(labels)
    sim = similarity_matrix(output)

    same = same_label(labels) # remove diagonal
    #diff = same*(-1) + 1  #turns 1's into 0's and 0's into 1's 
    
    same_M = (same * sim)
    #diff_M = (diff * sim)

    loss = - same_M.sum(dim = 1)/same.sum(dim = 1)
    
    return loss.sum() / ln
