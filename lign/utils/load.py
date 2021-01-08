from torchvision import datasets
import torch

def mnist_to_lign(path, transforms = None, train = True):
    from lign.graph import GraphDataset

    dataset =  datasets.MNIST(path, train=train)
    graph = GraphDataset()

    digits = []
    labels = []
    for img, lab in dataset:
        out = {
                "data": {},
                "edges": set()
            }

        if transforms:
            img = transforms(img)

        digits.append(img)
        labels.append(lab)
        graph.add(out)
    
    if(torch.is_tensor(digits[0])):
        digits = torch.stack(digits)
        labels = torch.LongTensor(labels)

    graph.set_data('x', digits)
    graph.set_data('labels', labels)

    return graph

def cifar_to_lign(path, transforms = None, train = True):
    from lign.graph import GraphDataset

    dataset =  datasets.CIFAR100(path, train=train)
    graph = GraphDataset()

    imgs = []
    labels = []
    for img, lab in dataset:
        out = {
                "data": {},
                "edges": set()
            }

        if transforms:
            img = transforms(img)

        imgs.append(img)
        labels.append(lab)
        graph.add(out)
    
    if(torch.is_tensor(imgs[0])):
        imgs = torch.stack(imgs)
        labels = torch.LongTensor(labels)

    graph.set_data('x', imgs)
    graph.set_data('labels', labels)

    return graph