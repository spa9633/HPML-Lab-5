import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import time


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

torch.manual_seed(43)
val_size = 15000
train_size = len(train) - val_size

train_ds, val_ds = random_split(train, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)

val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=True, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
net.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

valacc = 0
EPOCHS = 2000
t1 = time.time()
for epoch in range(EPOCHS):
    losses = []
    running_loss = 0
    for i, inp in enumerate(trainloader):
        inputs, labels = inp
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
    
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i%100 == 0 and i > 0:
            print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
            running_loss = 0.0

    avg_loss = sum(losses)/len(losses)
    scheduler.step(avg_loss)
    
    correct = 0
    total = 0

    if (epoch%20 == 0):
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = net(images)
        
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy on the validation set: ', 100*(correct/total), '%')
        if (100*(correct/total) >= 92):
            t2=time.time()
            valacc = 100*(correct/total)
            break
    
    if(valacc >= 92):
        break

print('Training and Validation Done - The time taken to reach 92% validation accuracy is', t2-t1, 'seconds')