import torch as th

def similarity_matrix(x, y, p = 2): #pairwise distance

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = th.pow(x - y, p).sum(2)
    
    return dist

def filter(data, labels, graph):
    fils = [lambda x: x == i for i in labels]

    out = graph.filter(fils, data)
    return out

class KNN():

    def __init__(self, X = None, Y = None, p = 2):
        self.train(X, Y)
        self.p = p

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if self.train_pts == None:
            raise RuntimeError("Knn wasn't trained. Need to execute self.train() first")
        
        dist = similarity_matrix(x, self.train_pts, self.p) ** (1/self.p)
        labels = th.argmin(dist, dim=1)
        return self.train_label[labels]

class Spectral(KNN):

    def __init__(self, X, Y, p = 2):
        super.__init__(X, Y, p)
        pass

    def predict(self):
        pass