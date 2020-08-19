
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


LAMBDA = 1e-3
DIST_VEC_SIZE = 2 # 3 was picked so the graph can be drawn in a 3d grid
INIT_NUM_LAB = 10
LABELS = np.arange(20)
SUBGRPAH_SIZE = 100
AMP_ENABLE = True
EPOCHS = 5
LR = 1e-3
RETRAIN_PER = {
    "superv": (0, 7),
    "semi": (0, 7)
}

np.random.shuffle(LABELS)

class LIGN_CIFAR(nn.Module):
    def __init__(self, out_feats):
        super(LIGN_CIFAR, self).__init__()
        self.gcn1 = md.layers.GCN(nn.Conv2d(3, 6, 5))
        self.gcn2 = md.layers.GCN(nn.Conv2d(6, 16, 5))
        self.gcn3 = md.layers.GCN(nn.Linear(16 * 5 * 5, 150))
        self.gcn4 = md.layers.GCN(nn.Linear(150, 84))
        self.gcn5 = md.layers.GCN(nn.Linear(84, out_feats))
        self.pool = md.layers.GCN(nn.MaxPool2d(2, 2))

    def forward(self, g, features):
        x = self.pool(g, F.relu(self.gcn1(g, features)))
        x = self.pool(g, F.relu(self.gcn2(g, x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.gcn3(g, x))
        x = F.relu(self.gcn4(g, x))
        
        return th.tanh(self.gcn5(g, x))

model = LIGN_CIFAR(DIST_VEC_SIZE).to(device)

accuracy = []
log = []
num_of_labels = len(LABELS)
opt = th.optim.Adam(model.parameters(), lr=LR)
scaler = GradScaler() if AMP_ENABLE else None

retrain_superv = lambda x: (x + RETRAIN_PER["superv"][0])%RETRAIN_PER["superv"][1] == 0
retrain_semi = lambda x: (x + RETRAIN_PER["semi"][0])%RETRAIN_PER["semi"][1] == 0


lg.train.superv(model, opt, dataset, "x", "labels", DIST_VEC_SIZE, LABELS[:INIT_NUM_LAB], LAMBDA, (device, scaler), addon = ADDON, subgraph_size=SUBGRPAH_SIZE, epochs=EPOCHS)

for num_labels in range(INIT_NUM_LAB, num_of_labels + 1):

    """if retrain_semi(num_labels):
        lg.train.semi_superv(model, opt, dataset, "x", "labels", DIST_VEC_SIZE, LABELS[:num_labels], LAMBDA, (device, scaler), addon = ADDON, subgraph_size=SUBGRPAH_SIZE, epochs=EPOCHS, cluster=(utl.clustering.NN(), 5))"""

    if retrain_superv(num_labels):
        lg.train.superv(model, opt, dataset, "x", "labels", DIST_VEC_SIZE, LABELS[:num_labels], LAMBDA, (device, scaler), epochs=EPOCHS, addon = ADDON, subgraph_size=SUBGRPAH_SIZE)
    
    acc = lg.test.accuracy(model, validate, dataset, "x", "labels", LABELS[:num_labels], cluster=(utl.clustering.NN(), 8), sv_img = ('2d', num_of_labels), device=device)

    accuracy.append(acc)
    log.append("Label: {}/{}\t|\tAccuracy: {}\t|\tSemisurpervised Retraining: {}\t|\t Surpervised Retraining: {}".format(num_labels, num_of_labels, round(acc, 2), retrain_semi(num_labels) == None, retrain_superv(num_labels)))
    print(log[-1])




### Save file
time = str(tm_now()).replace(":", "-").replace(".", "").replace(" ", "_")
filename = "LIGN_CIFAR_training_"+time

## Save metrics
metrics = {
    "accuracy": accuracy,
    "log": log
}
utl.io.json(metrics, "data/metrics/"+filename+".json")

## Save hyperparameters
para = {
    "LAMBDA": LAMBDA,
    "DIST_VEC_SIZE": DIST_VEC_SIZE,
    "INIT_NUM_LAB": INIT_NUM_LAB,
    "LABELS": LABELS.tolist(),
    "SUBGRPAH_SIZE": SUBGRPAH_SIZE,
    "AMP_ENABLE": AMP_ENABLE,
    "EPOCHS": EPOCHS,
    "LR": LR,
    "RETRAIN_PER": RETRAIN_PER
}

utl.io.json(para, "data/parameters/"+filename+".json")

## Save model
check = {
    "model": model.state_dict(),
    "optimizer": opt.state_dict()
}
if AMP_ENABLE:
    check["scaler"] = scaler.state_dict()

th.save(check, "data/models/"+filename+".pt")
