import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models

import SMD_opt
from models import resnet

num_epochs = 4500
num_classes = 10
batch_size = 128
learning_rate = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

model = ResNet18()

if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
model = model.cuda()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.01, 0.01)
        m.bias.data.uniform_(-0.1, 0.1)
    
model.apply(weights_init)

free_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(free_params)


criterion = nn.CrossEntropyLoss().cuda()
optimizer = SMD_opt.SMD_compress(model.parameters(), lr=learning_rate)

total_step = len(trainloader)
print(total_step)
loss_list = []
acc_list = []

# Training
for epoch in range(num_epochs):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(trainloader):
        
        images = images.cuda(async=True)
        labels = labels.cuda(async=True)
        
        images = Variable(images)
        labels = Variable(labels)
        # Run the forward pass
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and perform
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

         # Track the accuracy
        total = labels.size(0) + total
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item() + correct
        acc_list.append(correct / total)
        
        if (i + 1) % 100 == 0:
            w = torch.nn.utils.parameters_to_vector(model.parameters())
            print(w)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
    if (total == correct) :
        break 

print('Finished Training')


torch.save(model.state_dict(), './final_1norm.pth') 

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        images = images.cuda(async=True)
        labels = labels.cuda(async=True)
        
        images = Variable(images)
        labels = Variable(labels)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
