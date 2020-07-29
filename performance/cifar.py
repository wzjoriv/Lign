
import lign as lg
import lign.models as md
import lign.utils as utl

import torch as th
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

import numpy as np
import datetime
tm_now = datetime.datetime.now

dataset = lg.graph.GraphDataset("data/datasets/cifar100_train.lign")
validate = lg.graph.GraphDataset("data/datasets/cifar100_test.lign")

if th.cuda.is_available():
    device = th.device("cuda")
    th.cuda.empty_cache()
else:
    device = th.device("cpu")

def sum_neighs_data(neighs): ## adds up neighbors' data before executing post_mod (pre_mod happens before)
    out = neighs[0]
    for neigh in neighs[1:]:
        out = out + neigh
    return out

class ADDON(nn.Module): ## tempory layer for training
    def __init__(self, in_fea, out_fea):
        super(ADDON, self).__init__()
        self.gcn1 = md.layers.GCN(nn.Linear(in_fea, out_fea))
    
    def forward(self, g, features):
        x = self.gcn1(g, features)
        return x

LAMBDA = 0.001
DIST_VEC_SIZE = 2 # 3 was picked so the graph can be drawn in a 3d grid
INIT_NUM_LAB = 3
LABELS = np.arange(40)
SUBGRPAH_SIZE = 800
AMP_ENABLE = True
EPOCHS = 500
LR = 1e-3
RETRAIN_PER = {
    "superv": (0, 5),
    "semi": (0, 15)
}

np.random.shuffle(LABELS)

class LIGN_CIFAR(nn.Module):
    def __init__(self, out_feats):
        super(LIGN_CIFAR, self).__init__()
        self.gcn1 = md.layers.GCN(nn.Linear(32 * 32 * 3, 1000))
        self.gcn2 = md.layers.GCN(nn.Linear(1000, 300))
        self.gcn3 = md.layers.GCN(nn.Linear(300, 20))
        self.gcn4 = md.layers.GCN(nn.Linear(20, out_feats))

    def forward(self, g, features):
        x = features.view(-1, 32 * 32 * 3)
        x = F.relu(self.gcn1(g, x))
        x = F.relu(self.gcn2(g, x))
        x = self.gcn3(g, x)
        return th.tanh(self.gcn4(g, x))

model = LIGN_CIFAR(DIST_VEC_SIZE).to(device)

accuracy = []
log = []
num_of_labels = len(LABELS)
opt = th.optim.Adam(model.parameters(), lr=LR)
scaler = GradScaler() if AMP_ENABLE else None

retrain_superv = lambda x: x%RETRAIN_PER["superv"][1] == RETRAIN_PER["superv"][0]
retrain_semi = lambda x: x%RETRAIN_PER["semi"][1] == RETRAIN_PER["semi"][0]


lg.train.superv(model, opt, dataset, "x", "labels", DIST_VEC_SIZE, LABELS[:INIT_NUM_LAB], LAMBDA, (device, scaler), addon = ADDON, subgraph_size=SUBGRPAH_SIZE, epochs=EPOCHS)

for num_labels in range(INIT_NUM_LAB, num_of_labels + 1):

    """if retrain_semi(num_labels):
        lg.train.semi_superv(model, opt, dataset, "x", "labels", DIST_VEC_SIZE, LABELS[:num_labels], LAMBDA, (device, scaler), addon = ADDON, subgraph_size=SUBGRPAH_SIZE, epochs=EPOCHS, cluster=(utl.clustering.NN(), 5))"""

    if retrain_superv(num_labels):
        lg.train.superv(model, opt, dataset, "x", "labels", DIST_VEC_SIZE, LABELS[:num_labels], LAMBDA, (device, scaler), epochs=EPOCHS, addon = ADDON, subgraph_size=SUBGRPAH_SIZE)
    
    acc = lg.test.accuracy(model, validate, dataset, "x", "labels", LABELS[:num_labels], cluster=(utl.clustering.NN(), 5), sv_img = '2d', device=device)

    accuracy.append(acc)
    log.append("Label: {}/{}\t|\tAccuracy: {}\t|\tSemisurpervised Retraining: {}\t|\tSurpervised Retraining: {}".format(num_labels, num_of_labels, round(acc, 2), retrain_semi(num_labels), retrain_superv(num_labels)))
    print(log[-1])



