import torch as th
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

import numpy as np
import datetime
tm_now = datetime.datetime.now

transform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

trainset = tv.datasets.CIFAR100(root='../data/datasets/CIFAR100', train=True, transform=transform)
trainloader = th.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=2)

testset = tv.datasets.CIFAR100(root='../data/datasets/CIFAR100', train=False, transform=transform)
testloader = th.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

if th.cuda.is_available():
    device = th.device("cuda")
    th.cuda.empty_cache()
else:
    device = th.device("cpu")


AMP_ENABLE = True
EPOCHS = 50
LR = 1e-3

class Net(nn.Module):
    def __init__(self, out_fea):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_fea)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net(100).to(device)

accuracy = []
log = []

opt = th.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler() if AMP_ENABLE else None

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        opt.zero_grad()

        if AMP_ENABLE:
            with th.cuda.amp.autocast():
                out = model(inputs)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
        else:
            out = model(inputs)
            loss = criterion(out, labels)

            loss.backward()
            opt.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


time = str(tm_now()).replace(":", "-").replace(".", "").replace(" ", "_")
filename = "CIFAR_training_"+time


## Save model
check = {
    "model": model.state_dict(),
    "optimizer": opt.state_dict()
}
if AMP_ENABLE:
    check["scaler"] = scaler.state_dict()

th.save(check, "../data/models/"+filename+".pt")


correct = 0
total = 0
with th.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = th.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(100))
class_total = list(0. for i in range(100))
with th.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = th.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(100):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(100):
    print('Accuracy of %5s : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))