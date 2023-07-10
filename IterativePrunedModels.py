'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
import csv
import time

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
writeLines = []
sparsityRatios = [0.5,0.75,0.9]
for sparsityRatio in sparsityRatios:
  print("==> Peforming Iterative Pruning with Sparsity Ratio of: " + str(sparsityRatio))
  writeLines.append("==> Peforming Iterative Pruning with Sparsity Ratio of: " + str(sparsityRatio))
  # Model
  print('==> Building model..')
  writeLines.append('==> Building model..')
  net = ResNet18()
  net = net.to(device)
  if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True

  # Load checkpoint.
  print('==> Loading Pre-trained Model..')
  writeLines.append('==> Loading Pre-trained Model..')
  assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
  checkpoint = torch.load('./checkpoint/ckpt.pth')
  # Load pre-trained model that was able to achieve 93% test accuracy
  net.load_state_dict(checkpoint['net'])


  # Create list of modules and params that will be pruned in the future
  print("==> List Names of Layers and Corresponding Params..")
  writeLines.append("==> List Names of Layers and Corresponding Params..")
  parameters_to_prune = []
  layersNames = []
  for name, module in net.named_modules():
      # Print all conv2d layers in the resnet and the params they include
      if isinstance(module, torch.nn.Conv2d):
          print("Name of conv layer: " + name + " | Contained Params: ", end ="")
          for x in list(module.named_parameters()):
            temp = []
            print(x[0])
            temp.append(net.get_submodule(name))
            layersNames.append(name)
            temp.append(x[0])
            #append module and param to prune in the form of a tuple(modulename, param)
            parameters_to_prune.append(tuple(temp))

      # Print all linear layers in the resnet and the params they include
      elif isinstance(module, torch.nn.Linear):
          print("Name of linear layer: " + name + " | Contained Params: ", end ="")
          for x in list(module.named_parameters()):
            temp = []
            if x[0] == 'weight':
              print(x[0])
              temp.append(net.get_submodule(name))
              temp.append(x[0])
              layersNames.append(name)
              #append module and param to prune in the form of a tuple(modulename, param)
              parameters_to_prune.append(tuple(temp))
  print()
  #convert list to tuple form
  parameters_to_prune = tuple(parameters_to_prune)


  print("==> Perform Global Pruning..")
  writeLines.append("==> Perform Global Pruning..")
  globalSparsity = 0.0
  globalSparsityTotal = 100
  while (100. * globalSparsity / globalSparsityTotal) < (100. *sparsityRatio):
    #Perform a global prune
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=0.1,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training
    def train(epoch):
        start = time.time()
        print('\nEpoch: %d' % epoch)
        writeLines.append('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        end = time.time()
        print("Epoch Training Time: " + str(end - start) + " | " + 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        writeLines.append("Epoch Training Time: " + str(end - start) + " | " + 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return end - start

    def test(epoch):
        start = time.time()
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        end = time.time()
        print("Epoch Test Time: " + str(end - start) + " | " + 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        writeLines.append("Epoch Test Time: " + str(end - start) + " | " + 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    totaltrainTime = 0
    for epoch in range(0, 90):
        totaltrainTime += train(epoch)
        print("Running Total Training Time: " + str(totaltrainTime))
        test(epoch)
        scheduler.step()
    print("Total Time to Train: " + str(totaltrainTime))

    globalSparsity = 0.0
    globalSparsityTotal = 0.0
    #Calculate global sparsity
    for x in range(0,len(parameters_to_prune)):
      if parameters_to_prune[x][1] == 'weight':
        globalSparsity += float(torch.sum(parameters_to_prune[x][0].weight == 0))
        globalSparsityTotal += float(parameters_to_prune[x][0].weight.nelement())
      else:
        globalSparsity += float(torch.sum(parameters_to_prune[x][0].bias == 0))
        globalSparsityTotal += float(parameters_to_prune[x][0].bias.nelement())
    print("==> List Global Sparsity")
    writeLines.append("==> List Global Sparsity")
    print("Global sparsity: {:.2f}%".format(100. * globalSparsity / globalSparsityTotal))
    writeLines.append("Global sparsity: {:.2f}%".format(100. * globalSparsity / globalSparsityTotal))

with open("iterativeOutput.csv", "w", newline='') as f:
  writer = csv.writer(f)
  writer.writerows(writeLines)
  
