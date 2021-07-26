import torch as th

from lign.utils.functions import distance_matrix, randomize_tensor

class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p) ** (1/self.p)
        labels = th.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X = None, Y = None, k = 3, p = 2):
        self.k = k
        super().__init__(X, Y, p)
    
    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")
        
        dist = distance_matrix(x, self.train_pts, self.p) ** (1/self.p)

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = th.zeros(votes.size(0), dtype = self.train_label.dtype)
        count = th.zeros(votes.size(0), dtype = votes.dtype) - 1

        for lab in self.unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner


class KMeans(NN):

    def __init__(self, X = None, k=2, n_iters = 10, p = 2):

        self.k = k
        self.n_iters = n_iters
        self.p = p

        if type(X) != type(None):
            self.train(X)

    def train(self, X):

        self.train_pts = randomize_tensor(X)[:self.k]
        self.train_label = th.LongTensor(range(self.k))

        for _ in range(self.n_iters):
            labels = self.predict(X)

            for lab in range(self.k):
                select = labels == lab
                self.train_pts[lab] = th.mean(X[select], dim=0)

class Spectral(KNN):

    def __init__(self, X = None, k=2, n_iters = 10, p = 2):
        name = self.__class__.__name__
        raise NotImplementedError(f"{name} hasn't been implemented yet.")

