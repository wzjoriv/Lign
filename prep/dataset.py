from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json.load, json.dumps
import pickle.load, pickle.dump
import os.path

"""
    node = {
        data: {},
        edges: []
    }
    
"""

class GraphDataset(Dataset):
    def __init__(self, file="../utils/defaults/graph.lign", heavy=False, workers = 1, transform = (None, 'x')):
        self.dataset = None
        self.heavy = heavy
        self.workers = workers
        self.transform = transform

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

        if self.transform[0]:
            for data_x in self.transform[1:]:
                out["data"][data_x] = self.transform[0](data_x)

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

    def subgraph(self, nodes, edges = False, linked = True): #returns graph instance
        subgraph = SubGraph(self, nodes, edges, linked)
        return subgraph

    def pull(nodes=[]): #pulls others' data from nodes that it points to into it's temp
        pass

    def push(nodes=[]): #pushes its data to nodes that it points to into nodes's temp
        pass

    def apply(func, nodes=[]):
        pass

    def reset_temp(): #clear collected data from other nodes
        self.dataset["__temp__"] = [[] for i in range(self.dataset["count"])]

    def filter(func): #returns nodes that pass the filter
        pass

    def save(self, file="", heavy = False):

        if len(file):
            if self.heavy:
                pass
            else:
                pass
        
        pass

class SubGraph(): #creates a isolated graph from the dataset. Meant to be more efficient if only changing a few nodes from the dataset
    def __init__(self, graph_dataset, nodes, edges = False, linked = False):
        self.dataset = graph_dataset
        self.count = len(nodes)
        self.__temp__ = [[] for i in range(self.count)]
        self.__nodes__ = self.dataset[nodes]                             ### Need to fix
        
        if edges: 
            self.__edges__ = [[] for i in range(self.count)]
        if not linked: 
            self.__linked_nodes__ = self.__nodes__
            self.__nodes__ = [self.dataset[i].copy() for i in nodes]   ### Need to fix
    
    def __len__(self):
        return self.dataset["count"]

    def __getitem__(self, indx):
        pass

    def add(self, nodes):
        pass

    def pull(nodes=[]): #pulls others' data from nodes that it points to into it's temp
        pass

    def push(nodes=[]): #pushes its data to nodes that it points to into nodes's temp
        pass

    def apply(func, nodes=[]):
        pass

    def reset_temp(): #clear collected data from other nodes
        self.__temp__ = [[] for i in range(self.count)]

    def filter(func): #returns nodes that pass the filter
        pass

    def to_dataset(): #if not linked
        pass








"""
formats cheat sheet:
    (format[, folder/file1, folder/file2])                  ## size of data type in format must be the same as the number of directories/files

    syntax:
        - = addition entries in the data field
        (NAME) = give data the name NAME in the data field
        [##] = optional
            csv: [column1, column2, 3, [0_9]]               ##  Indicate index or column name to retrieve; multiple columns are merges as one

    data type:
        imgs = images folder                                ### Heavy lign graph suggested for large images
        csv = csv file
        imgs_url = file of list of images url                ### Heavy lign graph suggested

    example:
        format = ('imgs(x)-csv(label)[column2]', 'data/', 'labels.txt')
"""

def data_to_dataset(format, out_path, heavy = False):
    pass
    return out_path