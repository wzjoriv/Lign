# # Lign - MNIST
# 
# ----
# 
# ## Imports


import lign as lg
import lign.models as md
import lign.utils as utl

import torch as th
import torchvision as tv
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

import numpy as np
import datetime
tm_now = datetime.datetime.now


# ### Load Dataset

dataset_name = "MNIST" #<<<<<

dataset = lg.graph.Graph("data/datasets/mnist_train.lign")
validate = lg.graph.Graph("data/datasets/mnist_test.lign")


# ### Cuda GPUs


if th.cuda.is_available():
    device = th.device("cuda")
    th.cuda.empty_cache()
else:
    device = th.device("cpu")


# ### Hyperparameters
# * LAMBDA: regulates how much the model relies on difference between the nodes vs the features that lead to their label 
# * DIST_VEC_SIZE: size of vector representing the mapping of the nodes by the model
# * INIT_NUM_LAB: number of labels used to training the model initially in the supervised method to learn pairwise mapping
# * LABELS: list of all the labels that model comes across. Labels can be appended at any time. The order of labels is initially randomized
# * SUBGRAPH_SIZE: represent the number of nodes processed at once. The models don't have batches. This is the closest thing to it
# * AMP_ENABLE: toggle to enable mixed precission training
# * EPOCHS: Loops executed during training
# * LR: Learning rate
# * RETRAIN_PER: period between retraining based on number of labels seen. format: (offset, period)


LAMBDA = 0.1
DIST_VEC_SIZE = 128 #128
INIT_NUM_LAB = 4
LABELS = np.arange(10)
SUBGRPAH_SIZE = 500
AMP_ENABLE = True
EPOCHS = 300
LR = 1e-3
RETRAIN_PER = {
    "superv": (6, 3)
}

np.random.shuffle(LABELS)


# ---
# ## Models
# ### Lign
# 
# [L]ifelong Learning [I]nduced by [G]raph [N]eural Networks Model (Lign)

class LIGN_MNIST(nn.Module):
    def __init__(self, out_feats):
        super(LIGN_MNIST, self).__init__()
        self.gcn1 = lg.layers.GCN(nn.Conv2d(1, 32, 3, 1))
        self.gcn2 = lg.layers.GCN(nn.Conv2d(32, 64, 3, 1))
        self.gcn3 = lg.layers.GCN(nn.Linear(9216, out_feats))
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, g, features, save=False):
        x = F.relu(self.gcn1(g, features))
        if(save):
            x_sv = x[:100]
            print(x_sv.shape)
            for i in range(x_sv.size(0)):
                save_image(th.unsqueeze(x_sv[i], 1), "data/thesis/mid/gcn1/img" + str(i) + ".png")

        x = F.relu(self.gcn2(g, x))
        if(save):
            x_sv = x[:100]
            for i in range(x_sv.size(0)):
                save_image(th.unsqueeze(x_sv[i], 1), "data/thesis/mid/gcn2/img" + str(i) + ".png")
        x = F.max_pool2d(x, 2)
        x = th.flatten(self.drop1(x), 1)
        x = F.relu(self.gcn3(g, x))
        if(save):
            x_sv = x[:100]
            save_image(th.unsqueeze(th.unsqueeze(x_sv, 1), -1), "data/thesis/mid/vector output.png", nrow=100)

        return self.drop2(x)


class ADDON(nn.Module): ## tempory layer for training
    def __init__(self, in_fea, out_fea, device = 'cuda'):
        super(ADDON, self).__init__()
        self.addon = lg.layers.ADDON(in_fea, out_fea, device=device) #
    
    def forward(self, g, features):
        x = F.log_softmax(self.addon(g, features), dim=1)
        return x

model = LIGN_MNIST(DIST_VEC_SIZE).to(device) # base
addon = ADDON(DIST_VEC_SIZE, INIT_NUM_LAB, device).to(device) # classifier


# ----
# ## Training
# ### Parameters


#opt
accuracy = []
log = []
num_of_labels = len(LABELS)
opt = th.optim.Adam(model.parameters(), lr=LR)
scaler = GradScaler() if AMP_ENABLE else None

retrain_superv = lambda x: (x + RETRAIN_PER["superv"][0])%RETRAIN_PER["superv"][1] == 0


# ### Train Model


acc = lg.test.accuracy(model, validate, dataset, "x", "labels", LABELS[:INIT_NUM_LAB], cluster=(utl.clustering.NN(), 5), sv_img = '2d', device=device)

accuracy.append(acc)
log.append("Label: {}/{} -- Accuracy: {}% -- Original".format(INIT_NUM_LAB, num_of_labels, round(acc, 2)))
print(log[-1])


lg.train.superv(model, opt, dataset, "x", "labels", LABELS[:INIT_NUM_LAB], addon, LAMBDA, (device, scaler), epochs=EPOCHS, sub_graph_size=SUBGRPAH_SIZE)

for num_labels in range(INIT_NUM_LAB, num_of_labels + 1):
  
    if retrain_superv(num_labels):
        acc = lg.test.accuracy(model, validate, dataset, "x", "labels", LABELS[:num_labels], cluster=(utl.clustering.NN(), 5), device=device)

    	accuracy.append(acc)
        log.append("Label: {}/{} -- Accuracy: {}% -- Surpervised Retraining: {}".format(num_labels, num_of_labels, round(acc, 2), False))
        print(log[-1])

        addon.addon.update_size(num_labels)
        EPOCHS -= int(EPOCHS*0.05)
        lg.train.superv(model, opt, dataset, "x", "labels", LABELS[:num_labels], addon, LAMBDA, (device, scaler), epochs=EPOCHS, sub_graph_size=SUBGRPAH_SIZE)
    
    acc = lg.test.accuracy(model, validate, dataset, "x", "labels", LABELS[:num_labels], cluster=(utl.clustering.NN(), 5), sv_img = '2d', device=device)

    accuracy.append(acc)
    log.append("Label: {}/{} -- Accuracy: {}% -- Surpervised Retraining: {}".format(num_labels, num_of_labels, round(acc, 2), retrain_superv(num_labels)))
    print(log[-1])


# ### Save State

time = str(tm_now()).replace(":", "-").replace(".", "").replace(" ", "_")
filename = "LIGN_" + dataset_name + "_training_"+time

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
    "RETRAIN_PER": RETRAIN_PER,
    "STRUCTURE": str(model)
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