import torch as th

def similarity_matrix(x, y=None, p = 2): #pairwise distance of vectors

    y = y if y else x

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = th.pow(x - y, p).sum(2)
    
    return dist

def same_label(y): # return matrix of nodes that have same label
    s = y.size(0)
    y_expand = y.unsqueeze(0).expand(s, s)
    Y = y_expand.eq(y_expand.t())
    return Y

def onehot_encoding(data, labels): # onehot encoding
    final = (data == labels[0]) * 0
    for lab in range(1, len(labels)):
        final |= (data == labels[lab]) * lab
    
    return final

def sum_neighs_data(neighs):
    out = neighs[0]
    for neigh in neighs[1:]:
        out = out + neigh
    return out

""" def distance_loss(output, labels, Lambda=0.01):

    ln = len(labels)
    sim = similarity_matrix(output)

    same = same_label(labels)  # remove diagonal
    #diff = same*(-1) + 1  #turns 1's into 0's and 0's into 1's

    same_M = (same * sim)
    #diff_M = (diff * sim)

    loss = - same_M.sum(dim=1)/same.sum(dim=1)

    return loss.sum() / ln """
