from argparse import ArgumentParser
import time 
import yaml
from tqdm import tqdm 
import sys, os, os.path
from copy import deepcopy

import random 
random.seed(0)

import numpy as np 
np.random.seed(0)

import torch
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

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

# Set environment variable to ensure deterministic behavior w/ CUDA 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

## removing non-determinism 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

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

Section('checkpoint', 'for when training from a checkpoint').params(
    from_checkpoint=Param(bool, 'whether to train from a checkpoint', required=True),
    trial=Param(int, 'which trial number to grab checkpoint from', required=True)
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

    ## ensure reproducibility 
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    raw_data = os.path.expandvars(raw_data)
    print('data from:', raw_data)
    trainset = torchvision.datasets.CIFAR10(root=raw_data, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)

    testset = torchvision.datasets.CIFAR10(root=raw_data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes

# Model
@param('training.arch')
def construct_model(arch):
    print('==> Building model..')

    model = None
    if arch == 'resnet':
        model = ResNet18()
    elif arch == 'vgg':
        model = VGG('VGG11')
    elif arch == 'mobilenet':
        model = MobileNetV2()
    elif arch == 'efficientnet':
        model = EfficientNetB0()
    elif arch == 'regnet':
        model = RegNetX_200MF()

    # if device == 'cuda':
        # model = torch.nn.DataParallel(model)
        # cudnn.benchmark = True
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
    print('free_params:', free_params)

    # w = torch.nn.utils.parameters_to_vector(model.parameters())
    # print(w)
    return model

@param('training.lr')
@param('training.epochs')
@param('training.pnorm')
@param('checkpoint.from_checkpoint')
@param('checkpoint.trial')
def train(model, trainloader, testloader, log_file=sys.stdout, output_directory='./', lr=None, epochs=None, pnorm=None, from_checkpoint=False, trial=0):
    criterion = nn.CrossEntropyLoss().cuda()
    if pnorm < 2.0:
        optimizer = SMD_opt.SMD_compress(model.parameters(), lr=lr, eps=pnorm-1)
    else:
        optimizer = SMD_opt.SMD_qnorm(model.parameters(), lr=lr, q=pnorm)

    total_step = len(trainloader)
    print('total_step:', total_step)
    print("learning rate =", lr)
    print("pnorm =", pnorm)

    best_test_acc = 0

    start_epoch = 0
    if from_checkpoint:
        checkpoint = torch.load(f'{output_directory}/checkpoint.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        print(f"Training from epoch {start_epoch+1}; current best test acc = {best_test_acc}", file=log_file)

    # Training
    for epoch in tqdm(range(start_epoch, epochs)):
        model.train()

        total = 0
        correct = 0

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

            # Track the accuracy
            total = labels.size(0) + total
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item() + correct

            if (i + 1) % 100 == 0:
                w = torch.nn.utils.parameters_to_vector(model.parameters())
                print(w)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, epochs, i + 1, total_step, loss.item(),
                            (correct / total) * 100))
       
        print('epoch {}: {} correct out of {}, loss: {}'.format(epoch+1, correct, total, loss.item()), file=log_file)
        if (total == correct) :
            break

        if (epoch + 1) % 25 == 0:
            print("Checkpointing...", file=log_file)
            print(f"{total_correct} correct out of {total_num}", file=log_file)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_test_acc': best_test_acc,
            }, f'{output_directory}/checkpoint.pt')

    print('Finished Training')

def evaluate(model, loader):
    # Test the model
    model.eval()
    with torch.no_grad():
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

    return correct / total


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='CIFAR-10 training')
    config.augment_argparse(parser)

    # loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    if config['checkpoint.from_checkpoint']:
        print("checkpointing!")
        os.environ["OUTPUT"] = "$GROUP/smd-experiment/cifar10_StochasticMirrorDescent-long-run/" + str(config['checkpoint.trial'])

    output_dir = os.path.expandvars(config['data.output_directory'])
    print(output_dir)
    output_dir = os.path.expandvars(output_dir)
    print(output_dir)

    if not config['checkpoint.from_checkpoint']:
        print("new training!")
        os.makedirs(output_dir, exist_ok = True)

    start_time = time.time()
    trainloader, testloader, _ = make_dataloaders()
    print(f'Data Load time (min): {(time.time() - start_time) / 60:.2f}')
    
    model = construct_model()
    # save initial model so we can view weights later
    if not config['checkpoint.from_checkpoint']:
        torch.save(model, f'{output_dir}/init_model.pt')

    with open(f'{output_dir}/log.txt', 'a') as log_file:
        log_file.write('\n')
        train(model, trainloader, testloader, log_file, output_dir)

    test_acc = evaluate(model, testloader)
    accuracies = {'test': test_acc}
    print(f'test accuracy: {test_acc*100:.2f}%')
    acc_file = f'{output_dir}/accuracy.yaml'
    with open(acc_file, 'w') as file:
        yaml.dump(accuracies, file)

    torch.save(model.state_dict(), f'{output_dir}/final_model.pt') 

    print(f'Total time (min): {(time.time() - start_time) / 60:.2f}')
