"""
This code is based on the following two repositories
* https://github.com/google-research/augmix
* https://github.com/erichson/NoisyMixup
"""

import argparse
import os

import augmentations
import numpy as np

from src.cifar_models import preactwideresnet18, preactresnet18, wideresnet28


import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from src.noisy_mixup import mixup_criterion
from src.tools import get_lr
from aug_utils import *

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--arch', '-m', type=str, default='preactresnet18',
    choices=['preactresnet18', 'preactwideresnet18', 'wideresnet28'], help='Choose architecture.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 0)')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--train-batch-size', type=int, default=128, help='Batch size.')
parser.add_argument('--test-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-wd', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# AugMix options
parser.add_argument('--mixture-width', default=3, type=int, help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture-depth', default=-1, type=int, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
parser.add_argument('--no-jsd', '-nj', action='store_true', help='Turn off JSD consistency loss.')
parser.add_argument('--all-ops', '-all', action='store_true', help='Turn on all operations (+brightness,contrast,color,sharpness).')

# Noisy Feature Mixup options
parser.add_argument('--alpha', type=float, default=0.0, metavar='S', help='for mixup')
parser.add_argument('--manifold_mixup', type=int, default=0, metavar='S', help='manifold mixup (default: 0)')
parser.add_argument('--add_noise_level', type=float, default=0.0, metavar='S', help='level of additive noise')
parser.add_argument('--mult_noise_level', type=float, default=0.0, metavar='S', help='level of multiplicative noise')

args = parser.parse_args()

def train(net, train_loader, optimizer, scheduler):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  
  criterion = torch.nn.CrossEntropyLoss().cuda()
  
  for i, (images, targets) in enumerate(train_loader):
    optimizer.zero_grad()

    if args.no_jsd:
        images = images.cuda()
        targets = targets.cuda()
      
        if args.alpha == 0.0:   
            outputs = net(images)
        else:
            outputs, targets_a, targets_b, lam = net(images, targets=targets, mixup_alpha=args.alpha,
                                                      manifold_mixup=args.manifold_mixup,
                                                      add_noise_level=args.add_noise_level,
                                                      mult_noise_level=args.mult_noise_level)
        
        if args.alpha>0:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)      
    
    
    else:
      images_all = torch.cat(images, 0).cuda()
      targets = targets.cuda()      
      
      if args.alpha == 0.0:   
            logits_all = net(images_all)
      else:
            logits_all, targets_a, targets_b, lam = net(images_all, targets=targets, mixup_alpha=args.alpha,
                                                      manifold_mixup=args.manifold_mixup,
                                                      add_noise_level=args.add_noise_level,
                                                      mult_noise_level=args.mult_noise_level)
        
      if args.alpha>0:
          logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
          loss = mixup_criterion(criterion, logits_clean, targets_a, targets_b, lam)
      else:
          logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
          loss = criterion(logits_clean, targets)         
      
      # JSD Loss

      p_clean, p_aug1, p_aug2 = F.softmax(
          logits_clean, dim=1), F.softmax(
              logits_aug1, dim=1), F.softmax(
                  logits_aug2, dim=1)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
      p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
      loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
  return loss_ema     


def test(net, test_loader):
      """Evaluate network on given dataset."""
      net.eval()
      total_loss = 0.
      total_correct = 0
      with torch.no_grad():
        for images, targets in test_loader:
          images, targets = images.cuda(), targets.cuda()
          logits = net(images)
          loss = F.cross_entropy(logits, targets)
          pred = logits.data.max(1)[1]
          total_loss += float(loss.data)
          total_correct += pred.eq(targets.data).sum().item()
    
      return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)


def main():
      torch.manual_seed(args.seed)
      np.random.seed(args.seed)
      torch.cuda.manual_seed(args.seed)
      torch.cuda.manual_seed_all(args.seed)
    
      # Load datasets
      train_transform = transforms.Compose(
          [transforms.RandomHorizontalFlip(),
           transforms.RandomCrop(32, padding=4)])
      preprocess = transforms.Compose(
          [transforms.ToTensor(),
           transforms.Normalize([0.5] * 3, [0.5] * 3)])
      test_transform = preprocess
    
      if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(
            './data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(
            './data/cifar', train=False, transform=test_transform, download=True)
        num_classes = 10
      else:
        train_data = datasets.CIFAR100(
            './data/cifar', train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(
            './data/cifar', train=False, transform=test_transform, download=True)
        num_classes = 100
    
      train_data = AugMixDataset(train_data, preprocess, args.no_jsd, args)
      train_loader = torch.utils.data.DataLoader(
          train_data, batch_size=args.train_batch_size,
          shuffle=True, num_workers=4, pin_memory=True)
    
      test_loader = torch.utils.data.DataLoader(
          test_data, batch_size=args.test_batch_size,
          shuffle=False, num_workers=4, pin_memory=True)
    
      # Create model
      if args.arch == 'preactresnet18':
        net = preactresnet18(num_classes=num_classes)
      elif args.arch == 'preactwideresnet18':
        net = preactwideresnet18(num_classes=num_classes)
      elif args.arch == 'wideresnet28':
          net = wideresnet28(num_classes=num_classes)
    
      optimizer = torch.optim.SGD(net.parameters(),
          args.learning_rate, momentum=args.momentum,
          weight_decay=args.decay, nesterov=True)
    
      # Distribute model across all visible GPUs
      net = torch.nn.DataParallel(net).cuda()
      #cudnn.benchmark = True
    
      start_epoch = 0
    
      scheduler = torch.optim.lr_scheduler.LambdaLR(
          optimizer,
          lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
              step, args.epochs * len(train_loader),
              1,  # lr_lambda computes multiplicative factor
              1e-6 / args.learning_rate))
    
      best_acc = 0
      print('Beginning training from epoch:', start_epoch + 1)
      
      for epoch in range(start_epoch, args.epochs):
            
                train_loss_ema = train(net, train_loader, optimizer, scheduler)
                test_loss, test_acc = test(net, test_loader)
            
                is_best = test_acc > best_acc
                best_acc = max(test_acc, best_acc)
            
                if is_best:
                  DESTINATION_PATH = args.dataset + '_models/'
                  OUT_DIR = os.path.join(DESTINATION_PATH, f'best_arch_{args.arch}_alpha_{args.alpha}_manimixup_{args.manifold_mixup}_addn_{args.add_noise_level}_multn_{args.mult_noise_level}_seed_{args.seed}')
                  if not os.path.isdir(DESTINATION_PATH):
                            os.mkdir(DESTINATION_PATH)
                  torch.save(net, OUT_DIR+'.pt')            
            
                print(
                    'Epoch {0:3d} | Train Loss {1:.4f} |'
                    ' Test Accuracy {2:.2f}'
                    .format((epoch + 1), train_loss_ema, 100. * test_acc))    
                
      DESTINATION_PATH = args.dataset + '_models/'
      OUT_DIR = os.path.join(DESTINATION_PATH, f'arch_{args.model}_alpha_{args.alpha}_manimixup_{args.manifold_mixup}_addn_{args.add_noise_level}_multn_{args.mult_noise_level}_seed_{args.seed}')
      if not os.path.isdir(DESTINATION_PATH):
                os.mkdir(DESTINATION_PATH)
      torch.save(net, OUT_DIR+'.pt')


if __name__ == '__main__':
  main()
