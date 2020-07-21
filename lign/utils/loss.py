import torch

def similarity_matrix(x): #pairwise distance
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
    
    dist = torch.pow(x - y, 2).sum(2)
    
    return dist

def same_label(y):
    s = y.size(0)
    y_expand = y.unsqueeze(0).expand(s, s)
    Y = y_expand.eq(y_expand.t())
    return Y

def distance_loss(output, labels):
    """
    if nodes with the same label: 
    if nodes with different label: 
    """

    sim = similarity_matrix(output)
    same = same_label(labels)
    diff = same*(-1) + 1  #turns 1's to 0's and 0's to 1's 
    
    same_M = (same * sim)
    diff_M = (diff * sim)
    loss = torch.sqrt(torch.sum(same_M))/torch.sum(same) - torch.sqrt(torch.sum(diff_M))/torch.sum(diff)
    
    return loss