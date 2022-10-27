from argparse import ArgumentParser
import time 
import numpy as np
import yaml
from tqdm import tqdm 
import sys, os, os.path
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models

import SMD_opt
from models import *

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section

Section('training', 'Hyperparameters').params(
    arch=Param(str, 'CNN architecture to use', required=True),
    pnorm=Param(float, 'p-value to use in SMD', required=True),
    # lr_init=Param(float, 'The initial learning rate to use', required=True),
    lr=Param(float, 'The maximum learning rate to use', required=True),
    epochs=Param(int, 'Number of epochs to run for', required=True),
    # lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', required=True),
    batch_size=Param(int, 'Batch size', default=128),
    num_workers=Param(int, 'The number of workers', default=8),
)

Section('data', 'arguments for loading data').params(
    output_directory=Param(str, 'directory to save outputs', required=True),
    raw_data=Param(str, 'Where the raw data can be found', required=True)
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

# Data
@param('data.raw_data')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(raw_data=None, batch_size=None, num_workers=None):
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

    raw_data = os.path.expandvars(raw_data)
    trainset = torchvision.datasets.CIFAR10(root=raw_data, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=raw_data, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

# Model
@param('training.arch')
def construct_model(arch):
    print('==> Building model..')

    model = None
    if "arch" == 'resnet':
        model = ResNet18()
    else:
        model = MobileNetV2()

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
        
    # model.apply(weights_init)

    free_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('free_params:', free_params)

    w = torch.nn.utils.parameters_to_vector(model.parameters())
    print(w)
    return model

@param('training.lr')
@param('training.epochs')
@param('training.pnorm')
def train(model, trainloader, testloader, log_file=sys.stdout, output_directory='./', lr=None, epochs=None, pnorm=None):
    criterion = nn.CrossEntropyLoss().cuda()
    if pnorm < 2.0:
        optimizer = SMD_opt.SMD_compress(model.parameters(), lr=lr, eps=pnorm-1)
    else:
        optimizer = SMD_opt.SMD_qnorm(model.parameters(), lr=lr, q=pnorm)

    total_step = len(trainloader)
    print('total_step:', total_step)

    best_test_acc = 0

    # Training
    for epoch in tqdm(range(epochs)):
        model.train()

        for i, (images, labels) in enumerate(trainloader):
            
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            images = Variable(images)
            labels = Variable(labels)
            # Run the forward pass
            outputs = model(images)
            
            loss = criterion(outputs, labels)

            # Backprop and perform
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
       
        w = torch.nn.utils.parameters_to_vector(model.parameters())
        print(w)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1} loss: {loss.item():.4f}', file = log_file)
            train_acc, test_acc = evaluate(model, trainloader, testloader)
            print(f'Epoch {epoch+1} train acc: {train_acc:.4f}', file = log_file)
            print(f'Epoch {epoch+1} test acc: {test_acc:.4f}', file = log_file)
            log_file.flush()

            if test_acc > best_test_acc:
                print("Saving best model...", file=log_file)
                best_test_acc = test_acc
                torch.save(model, f'{output_directory}/best_model.pt')

            if train_acc == 1.0 :
                break

    print('Finished Training')

def evaluate(model, trainloader, testloader):
    # Test the model
    model.eval()
    acc = []
    with torch.no_grad():
        for loader in (trainloader, testloader):
            correct = 0
            total = 0
            for images, labels in loader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                
                images = Variable(images)
                labels = Variable(labels)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(correct, total)
            acc.append(correct / total)

    return acc[0], acc[1]


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='CIFAR-10 training')
    config.augment_argparse(parser)

    # loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    print(config)

    output_dir = os.path.expandvars(config['data.output_directory'])
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    trainloader, testloader, _ = make_dataloaders()
    print(f'Data Load time (min): {(time.time() - start_time) / 60:.2f}')
    
    model = construct_model()
    with open(f'{output_dir}/log.txt', 'w') as log_file:
        train(model, trainloader, testloader, log_file, output_dir)

    train_acc, test_acc = evaluate(model, trainloader, testloader)
    accuracies = {'train': train_acc, 'test': test_acc}
    print(f'train accuracy: {train_acc*100:.2f}%')
    print(f'test accuracy: {test_acc*100:.2f}%')
    acc_file = f'{output_dir}/accuracy.yaml'
    with open(acc_file, 'w') as file:
        yaml.dump(accuracies, file)

    torch.save(model.state_dict(), f'{output_dir}/final_model.pt') 

    print(f'Total time (min): {(time.time() - start_time) / 60:.2f}')
