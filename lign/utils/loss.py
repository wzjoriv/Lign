import torch as th

def similarity_matrix(x, p = 2): #pairwise distance
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    y = x
    y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    """

    n = x.size(0)
    d = x.size(1)

    y = x.unsqueeze(0).expand(n, n, d)
    x = x.unsqueeze(1).expand(n, n, d)
    
    dist = th.pow(x - y, p).sum(2)
    
    return dist

def same_label(y):
    s = y.size(0)
    y_expand = y.unsqueeze(0).expand(s, s)
    Y = y_expand.eq(y_expand.t())
    return Y

def distance_loss(output, labels):

    sim = similarity_matrix(output)
    same = same_label(labels)
    diff = same*(-1) + 1  #turns 1's into 0's and 0's into 1's 
    
    same_M = (same * sim)
    diff_M = (diff * sim)
    loss = th.sqrt(th.sum(same_M))/th.sum(same) - th.sum(diff_M)/th.sum(diff)
    
    return loss