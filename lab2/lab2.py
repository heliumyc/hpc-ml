import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

# models defined, ResNet18
# res net model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def main():

    # argument parser
    parser = argparse.ArgumentParser(description='cifar10 res18')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--workers', default=2, type=int, help='data loader workers number')
    parser.add_argument('--epoch', default=5, type=int, help='epoch to run')
    parser.add_argument('--data', default='./data', type=str, help='data path')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--gpu', action='store_true', help='use gpu or not')
    parser.add_argument('--disable_batch_norm', action='store_true', help='use batch_norm or not')
    # sgd sgd-nesterov adagrad adadelta adam
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer')
    args = parser.parse_args()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    data_path = args.data
    workers_num = args.workers
    max_epoch = args.epoch
    disable_batch_norm = args.disable_batch_norm
    lr = args.lr

    # transformer
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # size of 32*32, padding 4 px
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # dataloaders
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=workers_num)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=workers_num)


    # net model
    net = ResNet(BasicBlock, [2, 2, 2, 2]) # res18
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd' :
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'sgd-nesterov':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=lr, weight_decay=5e-4)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=lr, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)

    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    best_acc = 0  # best test accuracy
    

    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        data_load_time = 0
        train_time = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            tic = time.perf_counter()
            inputs, targets = inputs.to(device), targets.to(device)
            data_load_time += time.perf_counter() - tic

            tic = time.perf_counter()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_time += time.perf_counter() - tic

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return data_load_time, train_time

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        data_load_time = 0
        with torch.no_grad():
            total_tic = time.perf_counter()
            for batch_idx, (inputs, targets) in enumerate(testloader):
                tic = time.perf_counter()
                inputs, targets = inputs.to(device), targets.to(device)
                data_load_time += time.perf_counter() - tic

                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        return data_load_time

    # start training
    total_data_load_time = 0
    total_train_time = 0
    total_epoch_time = 0
    for epoch in range(0, max_epoch):

        total_tic = time.perf_counter()
        train_data_load_time, train_time = train(epoch)
        test_data_load_time = test(epoch)
        data_load_time = train_data_load_time + test_data_load_time
        epoch_time = time.perf_counter() - total_tic

        print('Data-loading time: %.3f | Training time: %.3f | Epoch running time: %.3f' %(data_load_time, train_time, epoch_time))

        total_data_load_time += data_load_time
        total_train_time += train_time
        total_epoch_time += epoch_time

    
    print('Total-Data-loading time: %.3f | Total-Training time: %.3f | Total-Epoch running time: %.3f' %(total_data_load_time, total_train_time, total_epoch_time))
if __name__ == "__main__":
    main()