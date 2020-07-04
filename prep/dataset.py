from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json.load, json.dumps
import pickle.load, pickle.dump
import os.path

"""
    nodes = {
        data: [],
        edges: []
    }
"""

class GraphDataset(Dataset):
    def __init__(self, file="../utils/defaults/graph.lign", heavy=False, workers = 1):
        self.dataset = None
        self.heavy = heavy
        self.workers = workers

        self.__folder__ = ""
        self.__files__ = {"file": ""}
        if file != "../utils/defaults/graph.lign":
            self.__folder__ = os.path.dirname(file)
            self.__files__["file"] = file

        if self.heavy:
            self.__files__["data"] = os.path.join(self.__folder__, ".data_LIGN/", "")
            self.__files__["edges"] = os.path.join(self.__folder__, ".edges_LIGN/", "")

        with open(file, "r") as read_file:
            self.dataset = json.load(read_file)

        if "count" not in self.dataset and "data" not in self.dataset and \
            "edges" not in self.dataset and "count" not in self.dataset:
            raise FileNotFoundError
    
    def __len__(self):
        return self.dataset["count"]

    def __getitem__(self, indx):
        out = {}

        if self.heavy:
            with open(self.dataset["data"][indx], "rb") as read_file:
                out["data"] = pickle.load(read_file)

            with open(self.dataset["edges"][indx], "rb") as read_file:
                out["edges"] = pickle.load(read_file)
            
        else:
            out["data"] = self.dataset["data"][indx]
            out["edges"] = self.dataset["edges"][indx]

        return out

    def add(self, nodes):
        tp = type(nodes)

        if tp != list or tp != tuple or tp != set:
            nodes = [nodes]

        for nd in nodes:
            nd["edges"].add(self.dataset["count"])

            if self.heavy:
                fl = str(self.dataset["count"]) + ".lign.dt"

                out = os.path.join(self.__files__["data"], fl)
                self.dataset["data"].append(out)
                with open(out, "w") as write_file:
                    pickle.dump(nd["data"], write_file)

                out = os.path.join(self.__files__["edges"], fl)
                self.dataset["edges"].append(out)
                with open(out, "w") as write_file:
                    pickle.dump(nd["edges"], write_file)
            else:
                self.dataset["data"].append(nd["data"])
                self.dataset["edges"].append(nd["edges"])
                
            self.dataset["__temp__"].append([])
            self.dataset["count"] += 1

    def subgraph(self, nodes, edges = False):
        pass

    def pull(nodes=[]): #pulls other's data from nodes that it points to into it's temp
        pass

    def push(nodes=[]): #pushes its data to nodes that it points to into nodes's temp
        pass

    def apply(func, nodes=[]):
        pass

    def reset_temp(): #clear collected data from other nodes
        self.dataset["__temp__"] = [[] for i in range(self.dataset["count"])]

    def filter(func): #returns nodes that pass the filter
        pass

    def save(self, file="data/graph.lign", heavy = False):
        pass

"""
formats cheatsheet:
    (format[, folder/file1, folder/file2])
    size of data type must be the same

    syntax:
        - = other data
        (NAME) = give data the name NAME in the data field
        [##] = options
            csv: [label1, label2, 3, 0-9]

    data type:
        imgs = images
        csv = csv file

    example:
        format = ('imgs-csv[label](labels)', 'data/', 'labels.txt')
"""

def data_to_dataset(format, out_path, heavy = False):
    pass