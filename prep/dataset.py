class GraphDataset(object):
    def __init__(self, file="graph.lign"):
        self.__nodes__ = {"loaded": set(), "count": 0}

    def add(self, labels):
        pass
    
    def __write_to_disk__(self, info={}):
        pass

    def __read_from_disk__(self, info=[]):
        pass

def csv_to_graph(in_path, out_path, format="lign"):
    pass