import torch as th

from lign.utils.functions import similarity_matrix

class NN():

    def __init__(self, X = None, Y = None, p = 2):
        self.p = p
        self.train_pts = None
        self.train_label = None
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if self.train_pts == None:
            raise RuntimeError("NN wasn't trained. Need to execute NN.train() first")
        
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

        knn = dist.topk(self.k, largest=False)
        votes = self.train_label[knn.indices]

        winner = th.zeros(votes.size(0), dtype = self.train_label.dtype)
        count = th.zeros(votes.size(0), dtype = votes.dtype) - 1

        unique_labels = self.train_label.unique()

        for lab in unique_labels:
            vote_count = (votes == lab).sum(1)
            who = vote_count >= count
            winner[who] = lab
            count[who] = vote_count[who]

        return winner


class Spectral(NN):

    def __init__(self, X, Y, p = 2):
        super().__init__(X, Y, p)
        raise NotImplementedError("Spectral Clustering not yet implemented")

    def train(self, X, Y):
        raise NotImplementedError("Spectral Clustering not yet implemented")

    def predict(self):
        raise NotImplementedError("Spectral Clustering not yet implemented")

