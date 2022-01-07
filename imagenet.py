"""
This code is based on the following two repositories
* https://github.com/google-research/augmix
* https://github.com/erichson/NoisyMixup
"""

import argparse
import os

import augmentations
import numpy as np

from src.imagenet_models import resnet50


import torch
import torch.nn.functional as F
from robustness.tools.folder import ImageFolder
from torchvision import transforms

from src.noisy_mixup import mixup_criterion, do_noisy_mixup
from src.tools import get_lr
from aug_utils import *
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

IMAGENET_TRAIN_FOLDER = '/home/ubuntu/data/imagenet12/train'
IMAGENET_TEST_FOLDER = '/home/ubuntu/data/imagenet12/val'

parser = argparse.ArgumentParser(description='Trains an ImageNet Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--arch', '-m', type=str, default='resnet50',
    choices=['resnet50'], help='Choose architecture.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 0)')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--train-batch-size', type=int, default=256, help='Batch size.')
parser.add_argument('--test-batch-size', type=int, default=256)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-wd', type=float, default=0.0001, help='Weight decay (L2 penalty).')

# AugMix options
parser.add_argument('--augmix', type=int, default=1, metavar='S', help='aug mixup (default: 1)')
parser.add_argument('--mixture-width', default=3, type=int, help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture-depth', default=-1, type=int, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
parser.add_argument('--jsd', type=int, default=1, metavar='S', help='JSD consistency loss (default: 1)')

parser.add_argument('--all-ops', '-all', action='store_true', help='Turn on all operations (+brightness,contrast,color,sharpness).')

# Noisy Feature Mixup options
parser.add_argument('--alpha', type=float, default=0.0, metavar='S', help='for mixup')
parser.add_argument('--manifold_mixup', type=int, default=0, metavar='S', help='manifold mixup (default: 0)')
parser.add_argument('--add_noise_level', type=float, default=0.0, metavar='S', help='level of additive noise')
parser.add_argument('--mult_noise_level', type=float, default=0.0, metavar='S', help='level of multiplicative noise')
parser.add_argument('--sparse_level', type=float, default=0.0, metavar='S', help='sparse noise')

args = parser.parse_args()

def train(net, train_loader, optimizer, scheduler):
    """Train for one epoch."""
    net.train()
    loss_ema = 0.
  
    criterion = torch.nn.CrossEntropyLoss().cuda()
  
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, targets) in iterator:
        optimizer.zero_grad()
    
        k = 0 if args.alpha > 0.0 else -1
        if args.alpha > 0.0 and args.manifold_mixup == True: 
            k = np.random.choice(range(4), 1)[0]

        if args.jsd == 0:
            images = images.cuda()
            targets = targets.cuda()
        
            # intermediate results
            if k == 0:
                outputs = images
            else:
                outputs = net(images, resume_layer=0, exit_layer=k)

            # mix outputs
            if k >= 0:
                mixed_up_outputs, targets_a, targets_b, lam = do_noisy_mixup(outputs, targets, 
                                                                    jsd=args.jsd, 
                                                                    alpha=args.alpha, 
                                                                    add_noise_level=args.add_noise_level, 
                                                                    mult_noise_level=args.mult_noise_level,
                                                                    sparse_level=args.sparse_level)
                outputs = net(mixed_up_outputs, resume_layer=k+1, exit_layer=-1)
            
            if args.alpha>0:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)      
        
    
        elif args.jsd == 1:
            images_all = torch.cat(images, 0).cuda()
            targets = targets.cuda()      
        
            # intermediate results
            if k == 0:
                logits_all = images_all
            else:
                logits_all = net(images_all, resume_layer=0, exit_layer=k)

            # mix outputs
            if k >= 0:
                mixed_up_outputs, targets_a, targets_b, lam = do_noisy_mixup(logits_all, targets, 
                                                                    jsd=args.jsd, 
                                                                    alpha=args.alpha, 
                                                                    add_noise_level=args.add_noise_level, 
                                                                    mult_noise_level=args.mult_noise_level,
                                                                    sparse_level=args.sparse_level)
                logits_all = net(mixed_up_outputs, resume_layer=k+1, exit_layer=-1)
            
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
        loss_ema = loss_ema * 0.9 + float(loss.detach()) * 0.1
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
      
    augmentations.IMAGE_SIZE=224

      
    # Load datasets
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()])
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        preprocess,
    ])
         

    if args.dataset == 'imagenet':
        train_data = ImageFolder(root=IMAGENET_TRAIN_FOLDER, transform=train_transform)
        test_data = ImageFolder(root=IMAGENET_TEST_FOLDER, transform=test_transform)
        num_classes = 1000
    
    if args.augmix == 1:
        train_data = AugMixDataset(train_data, preprocess, args.jsd, args)
      
    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.train_batch_size,
            shuffle=True, num_workers=4, pin_memory=True)          
    
    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.test_batch_size,
            shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    if args.arch == 'resnet50':
        net = resnet50(num_classes=num_classes)
    
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
      
    for epoch in range(start_epoch, args.epochs):
            
        train_loss_ema = train(net, train_loader, optimizer, scheduler)
        test_loss, test_acc = test(net, test_loader)
            
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
            
        if is_best:
            DESTINATION_PATH = args.dataset + '_models/'
            OUT_DIR = os.path.join(DESTINATION_PATH, f'best_arch_{args.arch}_augmix_{args.augmix}_jsd_{args.jsd}_alpha_{args.alpha}_manimixup_{args.manifold_mixup}_addn_{args.add_noise_level}_multn_{args.mult_noise_level}_seed_{args.seed}')
            if not os.path.isdir(DESTINATION_PATH):
                os.mkdir(DESTINATION_PATH)
                torch.save(net, OUT_DIR+'.pt')            
            
            print('Epoch {0:3d} | Train Loss {1:.4f} |'
                ' Test Accuracy {2:.2f}'
                .format((epoch + 1), train_loss_ema, 100. * test_acc))    
                
        DESTINATION_PATH = args.dataset + '_models/'
        OUT_DIR = os.path.join(DESTINATION_PATH, f'final_arch_{args.arch}_augmix_{args.augmix}_jsd_{args.jsd}_alpha_{args.alpha}_manimixup_{args.manifold_mixup}_addn_{args.add_noise_level}_multn_{args.mult_noise_level}_seed_{args.seed}')
        if not os.path.isdir(DESTINATION_PATH):
            os.mkdir(DESTINATION_PATH)
        torch.save(net, OUT_DIR+'.pt')


if __name__ == '__main__':
    main()
