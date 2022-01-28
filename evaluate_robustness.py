import argparse
import os

import numpy as np
import torch
import torch.optim
import torch.utils.data
import collections

from src.get_data import NOISE_TYPES, SEVERITIES
from src.get_data import getData

from torch import nn
from torch.nn import functional as F

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

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def get_calibration(data_loader, model, debug=False, n_bins=None):
    model.eval()
    mean_conf, mean_acc = [], []
    ece = []

    # New code to compute ECE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ------------------------------------------------------------------------
    logits_list = []
    labels_list = []
    #mean_conf, mean_acc = [], []

    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)  # TODO: one hot or class number? We want class number
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(target)

    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()

    ece_criterion = _ECELoss(n_bins=15).cuda()
    ece = ece_criterion(logits, labels).item()
    #print('ECE (new implementation):', ece * 100)

    #calibration_dict = _get_calibration(labels, torch.nn.functional.softmax(logits, dim=1), debug=False, num_bins=n_bins)
    #mean_conf = calibration_dict['reliability_diag'][0]
    #mean_acc = calibration_dict['reliability_diag'][1]
    #print(mean_conf.shape)
    
    #calibration_results = {'reliability_diag': (torch.vstack(mean_conf), torch.vstack(mean_acc)), 'ece': ece}
    #calibration_results = {'reliability_diag': ((mean_conf), (mean_acc)), 'ece': ece}

    return ece


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
        ece = get_calibration(test_loader_clean, model, debug=False)
        print('Calibration: ', ece* 100)
        
        # Robust Accuracy
        rob_test_acc = []
        ece = []
        for noise in NOISE_TYPES:
            results[m][noise] = collections.defaultdict(dict)
            
            temp_results = []
            for severity in SEVERITIES:
                _, test_loader = getData(name=datasetc, train_bs=128, test_bs=1024, severity=severity, noise=noise)
                result_m = cls_validate(test_loader, model)
                results[m][noise][severity] = result_m
                rob_test_acc.append(result_m)
                ece.append( get_calibration(test_loader, model, debug=False))
                temp_results.append(result_m)
            
            print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(noise, 100 * np.mean(temp_results)))

            print(temp_results)
            
            
        with open(f"{save_dir}/{datasetc}/robust_{m}.pickle", "wb") as f:
            np.save(f, result_m)

        print('***')
        print('Average Robust Accuracy: ', np.mean(rob_test_acc))
        print('ECE (%): {:.2f}'.format(np.mean(ece)* 100))
        print('***')
        
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


