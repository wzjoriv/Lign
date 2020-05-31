__record_labels__ = [] #tracks labels used thus far

def complete(model, dataset, order=[], labels=[]):
    pass

def initial(model, dataset, labels=[]):
    pass

def retrain(model, dataset, new_labels=[], labels=[]):

    if len(new_labels) and len(labels):
        pass
    elif len(new_labels) and not len(labels):
        pass
    elif not len(new_labels) and len(labels):
        pass
    else:
        raise ValueError("parameters new_labels or labels must be given")