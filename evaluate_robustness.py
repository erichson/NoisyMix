import argparse
import os

import numpy as np
import torch
import torch.optim
import torch.utils.data
import collections

from src.get_data import NOISE_TYPES, SEVERITIES
from src.get_data import getData


def cls_validate(val_loader, model, time_begin=None):
    model.eval()
    acc1_val = 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(images)
            model_logits = output[0] if (type(output) is tuple) else output
            pred = model_logits.data.max(1, keepdim=True)[1]
            acc1_val += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            n += len(images)
    avg_acc1 = (acc1_val / n)
    return avg_acc1

def evaluate(folder, dataset, save_dir):

    datasetc = dataset + str("c")
    os.makedirs(os.path.join(save_dir, datasetc), exist_ok=True)

    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    print(models)

    results = collections.defaultdict(dict)
    _, test_loader_clean = getData(name=dataset, train_bs=128, test_bs=1024)

    for index, m in enumerate(models):
        model = torch.load(folder + m)
        print(m)
        model.eval()
        
        # Clean Accuracy
        clean_test_acc = cls_validate(test_loader_clean, model, time_begin=None)        
        print('Test Accuracy: ', clean_test_acc)
        
        # Robust Accuracy
        rob_test_acc = []
        for noise in NOISE_TYPES:
            results[m][noise] = collections.defaultdict(dict)
            for severity in SEVERITIES:
                _, test_loader = getData(name=datasetc, train_bs=128, test_bs=1024, severity=severity, noise=noise)
                result_m = cls_validate(test_loader, model)
                results[m][noise][severity] = result_m
                rob_test_acc.append(result_m)
                
        with open(f"{save_dir}/{datasetc}/robust_{m}.pickle", "wb") as f:
            np.save(f, result_m)

        
        print('Average Robust Accuracy: ', np.mean(rob_test_acc))


    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser("MegaMix")
    parser.add_argument("--dir", type=str, default='cifar10_models/', required=False, help='model dir')
    parser.add_argument("--dataset", type=str, default='cifar10', required=False, help='dataset')
    parser.add_argument("--batch_size", default=1000, type=int, help='batch size')
    args = parser.parse_args()

    test_batch_size = args.batch_size
    os.makedirs('eval_results', exist_ok=True)
    evaluate(args.dir, args.dataset, 'eval_results')


