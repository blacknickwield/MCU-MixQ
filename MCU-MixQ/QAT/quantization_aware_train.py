import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm
import argparse
from model import MixQTinyVGG

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--step-epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--w_bits', type=str, default='2345678')
    parser.add_argument('--a_bits', type=str, default='2345678')
    parser.add_argument('--share_weight', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lra', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--complexity-decay', type=float, default=0)
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--path', type=str)

    return parser.parse_args()

train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=test_transform
)


def construct_model(args) -> nn.Module:
    w_bits = [int(bit) for bit in args.w_bits]
    a_bits = [int(bit) for bit in args.a_bits]
    share_weight = args.share_weight
    cfg = None
    in_channels = args.in_channels
    num_classes = args.num_classes
    batch_norm = args.batch_norm

    model = MixQTinyVGG(
        w_bits=w_bits, a_bits=a_bits, share_weight=share_weight,
        cfg=cfg, in_channels=in_channels, num_classes=num_classes, batch_norm=batch_norm,
    )
    return model

def adjust_lr(epoch: int, step_epoch: int, acc_optim: torch.optim.Optimizer, arch_optim: torch.optim.Optimizer, lr: float, lra: float):
    lr = lr * (0.1 ** (epoch // step_epoch))
    lra = lra * (0.1 ** (epoch // step_epoch))

    for param_group in acc_optim.param_groups:
        param_group['lr'] = lr

    for param_group in arch_optim.param_groups:
        param_group['lr'] = lra

def search(model: nn.Module, args):
    batch_size = args.batch_size
    epochs, step_epoch = args.epochs, args.step_epoch
    lr, lra, momentum = args.lr, args.lra, args.momentum
    weight_decay, complexity_decay = args.weight_decay, args.complexity_decay

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    criterion = F.cross_entropy

    params = [param for name, param in model.named_parameters() if 'alpha' not in name]
    param_names = [name for name, _ in model.named_parameters() if 'alpha' not in name]
    arch_params = [param for name, param in model.named_parameters() if 'alpha' in name]
    arch_param_names = [name for name, _ in model.named_parameters() if 'alpha' in name]
    acc_optim: torch.optim.Optimizer = torch.optim.SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    arch_optim: torch.optim.Optimizer = torch.optim.SGD(params=arch_params, lr=lra, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(epochs):
        adjust_lr(epoch=epoch, step_epoch=step_epoch, acc_optim=acc_optim, arch_optim=arch_optim, lr=lr, lra=lra)
        train(model=model, train_loader=train_dataloader, criterion=criterion, acc_optim=acc_optim, arch_optim=arch_optim)
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        
            
def train(model: nn.Module, train_loader, criterion, acc_optim: torch.optim.Optimizer, arch_optim: torch.optim.Optimizer):
    model = model.train()
    for index, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        acc_loss = criterion(outputs, labels)
        complexity_loss = model.complexity_loss()
        loss = acc_loss + complexity_loss
        
        acc_optim.zero_grad()
        arch_optim.zero_grad()
        loss.backward()
        acc_optim.step()
        arch_optim.step()

if __name__ == '__main__':
    args = parse()
    model = construct_model(args=args)
    search(model=model, args=args)
    
    

    
    