import torch as th

def get_filter(i):
    return lambda x: x == i

def filter(data, labels, graph):
    fils = [get_filter(i) for i in labels]

    out = graph.filter(fils, data)
    return out

def filter_k(data, labels, graph, k = 3):
    out = []
    labs = []

    for label in labels:
        labs.extend([label] * k)
        out.extend(graph.filter(get_filter(label), data)[:k])

    return th.LongTensor(out), th.LongTensor(labs)


def similarity_matrix(x, y, p = 2): #pairwise distance

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    dist = th.pow(x - y, p).sum(2)
    
    return dist

class NN():

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
            raise RuntimeError("NN wasn't trained. Need to execute self.train() first")
        
        dist = similarity_matrix(x, self.train_pts, self.p) ** (1/self.p)
        labels = th.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X = None, Y = None, p = 2, k = 3):
        super().__init__(X, Y, p)
        self.k = k

    def predict(self, x):
        if self.train_pts == None:
            raise RuntimeError("KNN wasn't trained. Need to execute self.train() first")
        
        dist = similarity_matrix(x, self.train_pts, self.p) ** (1/self.p)

        votes = dist.argsort(dim=1)[:,:self.k]
        votes = self.train_label[votes]

        print(votes)
        print(th.unique(votes, dim = 1, return_counts=True))
        
        #max_count = count.argmax(dim=1)
        return votes[10]


class Spectral(NN):

    def __init__(self, X, Y, p = 2):
        super().__init__(X, Y, p)
        pass

    def predict(self):
        pass

if __name__ == '__main__':
    a = th.Tensor([
        [1, 1],
        [0.88, 0.90],
        [-1, -1],
        [-1, -0.88]
    ])

    b = th.LongTensor([3, 3, 5, 5])

    c = th.Tensor([
        [-0.5, -0.5],
        [0.88, 0.88]
    ])

    knn = KNN(a, b)
    print(knn(c))
