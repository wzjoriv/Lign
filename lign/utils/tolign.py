from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from ..graph import GraphDataset

def cal_to_lign(path):
    pass

def cifar_to_lign(path):
    dataset =  datasets.CIFAR100(path)
    graph = GraphDataset()

    for img, lab in dataset:
        out = {
                "data": {
                    "x": img,
                    "true_label": lab
                },
                "edges": []
            }
        graph.add(out)

    return graph

def core_to_lign(path):
    pass