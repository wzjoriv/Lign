import lign as lg
import lign.utils as utl

import torch as th
import torchvision as tv
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

import numpy as np
import datetime, os
from matplotlib.pyplot import plot as plt
tm_now = datetime.datetime.now

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

dataset_name = "CIFAR" #<<<<<

trans = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ### Load Dataset

dataset_name = "CIFAR" #<<<<<

dataset = lg.graph.Graph("data/datasets/cifar.lign")

split = 5/6
split_n = int(len(dataset)*split)
nodes_n = list(range(len(dataset)))

dataset_train = dataset.sub_graph(nodes_n[:split_n], get_data = True, get_edges = True)
dataset_validate = dataset.sub_graph(nodes_n[split_n:], get_data = True, get_edges = True)

# ### Cuda GPUs

if th.cuda.is_available():
    device = th.device("cuda")
    th.cuda.empty_cache()
else:
    device = th.device("cpu")

# ### Hyperparameters

LAMBDA = 0.05
DIST_VEC_SIZE = 300 #128
INIT_NUM_LAB = 50
LABELS = np.arange(100)
SUBGRPAH_SIZE = 128
AMP_ENABLE = True and th.cuda.is_available()
EPOCHS = 200
LR = 1e-3
RETRAIN_PER = { # (offset, frequency); When zero, true
    "superv": lambda x: not (x + 50)%10, 
    #"semi": lambda x: not (x + 1)%10,
    "semi": lambda x: False,
    #"unsuperv": lambda x: not (x + 5)%10
    "unsuperv": lambda x: False
}
ACCURACY_MED = utl.clustering.KNN()
LOSS_FUN = nn.CrossEntropyLoss()
STEP_SIZE = 1

num_of_labels = len(LABELS)
np.random.shuffle(LABELS)
t_methods = [lg.train.superv, lg.train.semi_superv, lg.train.unsuperv]
t_names = ["supervised", "semi-supervised", "unsupervised"]
scaler = GradScaler() if AMP_ENABLE else None
accuracy = []
log = []
label_and_acc = [[], []]
introductions = range(INIT_NUM_LAB + 1, num_of_labels + 1, STEP_SIZE) #start, end, step

# ---
# ## Models
# ### LIGN
# 
# [L]ifelong Learning [I]nduced by [G]raph [N]eural Networks Model (LIGN)

### CIFAR
from lign.models import CIFAR

base = CIFAR.Base(DIST_VEC_SIZE).to(device)  # base
classifier = CIFAR.Classifier(DIST_VEC_SIZE, INIT_NUM_LAB, device).to(device) # classifer

# ----
# ## Training

opt = th.optim.Adam([ # optimizer for the full network
        {'params': base.parameters()},
        {'params': classifier.parameters()}
    ], lr=LR)

def test_and_log(num_labels, text, method=utl.clustering.NN()):
    acc = lg.test.accuracy(base, 
                    dataset_validate, 
                    dataset_train, 
                    "x", "labels", 
                    LABELS[:num_labels], 
                    cluster=(method, 5), 
                    device=device)

                    
                    
    accuracy.append(acc)
    m_name = method.__class__.__name__
    log.append(f"Label: {num_labels}/{num_of_labels} -- Accuracy({m_name}): {round(acc, 2)}% -- {text}")
    label_and_acc[0].append(num_labels)
    label_and_acc[1].append(acc)
    print(log[-1])

# ### Train Model


# original network
test_and_log(INIT_NUM_LAB, "Original", method=ACCURACY_MED)

lg.train.superv(models = (base, classifier),
                        labels = LABELS[:INIT_NUM_LAB],
                        graph = dataset_train,
                        opt = opt,
                        tags = ("x", "labels"),
                        device = (device, scaler),
                        lossF = LOSS_FUN,
                        epochs=EPOCHS, 
                        sub_graph_size=SUBGRPAH_SIZE)

# trained network
test_and_log(INIT_NUM_LAB, "Initial training", method=ACCURACY_MED)

# online learning system
for num_labels in introductions:

    to_train = [RETRAIN_PER[t](num_labels) for t in ("superv", "semi", "unsuperv")]

    if sum(to_train):
        classifier.DyLinear.update_size(num_labels)
        opt = th.optim.Adam([ # optimizer for the full network
                {'params': base.parameters()},
                {'params': classifier.parameters()}
            ], lr=LR)
        
        EPOCHS -= int(EPOCHS*LAMBDA)

        for i in (k for k, en in enumerate(to_train) if en):

            test_and_log(num_labels, f"Before {t_names[i]} retraining", method=ACCURACY_MED)

            t_methods[i](models = (base, classifier),
                        labels = LABELS[:num_labels],
                        graph = dataset_train,
                        opt = opt,
                        tags = ("x", "labels"),
                        device = (device, scaler),
                        lossF = LOSS_FUN,
                        epochs=EPOCHS, 
                        sub_graph_size=SUBGRPAH_SIZE)
    
    test_and_log(num_labels, "Tested with labels " + str(LABELS[:num_labels]), method=ACCURACY_MED)

# ### Save State

time = str(tm_now()).replace(":", "-").replace(".", "").replace(" ", "_")
filename = "LIGN_" + dataset_name + "_training_"+time

## Save metrics
metrics = {
    "accuracy": accuracy,
    "log": log,
    "label_and_acc": label_and_acc
}
utl.io.json(metrics, os.path.join("data", "log", filename+".json"))

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
    "STRUCTURE": {
        "base": str(base),
        "classifier": str(classifier)
    }
}

utl.io.json(para, os.path.join("data", "parameters", filename+".json"))

## Save model
check = {
    "base": base.state_dict(),
    "classifier": classifier.state_dict(),
    "optimizer": opt.state_dict()
}
if AMP_ENABLE:
    check["scaler"] = scaler.state_dict()

th.save(check, os.path.join("data", "models", filename+".pt"))