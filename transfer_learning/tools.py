import robustbench
import torch
from robustness import defaults, tools
from robustness.defaults import add_args_to_parser
import argparse
import cox
import numpy as np

import sys
import os
from pathlib import Path
this_scripts_path = Path(os.path.abspath(__file__))
sys.path.insert(1, str(this_scripts_path.parent.parent))
from src.imagenet_models import resnet50
import transfer_learning.constants as constants
import transfer_learning.datasets as datasets



def make_parser():
    parser = initialize_parser()
    parser = add_defaults_to_parser(parser)
    parser = add_custom_args_to_parser(parser)
    return parser


def initialize_parser():
    return argparse.ArgumentParser(description='Transfer learning via pretrained Imagenet models',
                                 conflict_handler='resolve')


def add_defaults_to_parser(parser):
    parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
    parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
    parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
    parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)
    return parser


def check_and_fill_config_training_and_model_loader_args(args):
    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, None)
    args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, None)
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, None)
    # args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, None)
    return args


def add_custom_args_to_parser(parser):
    parser.add_argument('--dataset', type=str, default='cifar10', choices=list(datasets.DS_TO_FUNC.keys()),
                        help='Dataset (Overrides the one in robustness.defaults)')
    parser.add_argument('--freeze-level', type=int, default=-1,
                        help='Up to what layer to freeze in the pretrained model (assumes a resnet architectures)')
    parser.add_argument('--per-class-accuracy', action='store_true', help='Report the per-class accuracy. '
                        'Can be used only with pets, caltech101, caltech256, aircraft, and flowers.')
    parser.add_argument('--model_id', type=str, default='megamix', choices=get_supported_model_ids(), 
                        help='Report the per-class accuracy. ')
    parser.add_argument('--out_dir', type=str, default='results', 
                        help='Where the results are saved. ')
    return parser


def check_that_dataset_supports_per_class_accuracy(args):
    if args.per_class_accuracy:
        assert args.dataset in ['pets', 'caltech101', 'caltech256', 'flowers', 'aircraft'], \
            f'Per-class accuracy not supported for the {args.dataset} dataset.'


def make_store(args):
    store = initialize_store(args.out_dir, f'{args.model_id}_{args.dataset}_{args.freeze_level}') 
    store = add_metadata_to_store_if_possible(store, args)
    return store


def initialize_store(out_dir, exp_name):
    return cox.store.Store(out_dir, exp_name)


def add_metadata_to_store_if_possible(store, args):
    if 'metadata' not in store.keys:
        args_dict = args.__dict__
        schema = cox.store.schema_from_dict(args_dict)
        store.add_table('metadata', schema)
        store['metadata'].append_row(args_dict)
    else:
        print('[Found existing metadata in store. Skipping this part.]')
    return store


def modify_args_if_using_per_class_accuracy(args, validation_loader):
    if args.per_class_accuracy:
        args.custom_accuracy = get_per_class_accuracy(args, validation_loader)
    return args


def get_per_class_accuracy(args, loader):
    '''Returns the custom per_class_accuracy function. When using this custom function         
    look at only the validation accuracy. Ignore trainig set accuracy.
    '''
    def _get_class_weights(args, loader):
        '''Returns the distribution of classes in a given dataset.
        '''
        if args.dataset in ['pets', 'flowers']:
            targets = loader.dataset.targets

        elif args.dataset in ['caltech101', 'caltech256']:
            targets = np.array([loader.dataset.ds.dataset.y[idx]
                                for idx in loader.dataset.ds.indices])

        elif args.dataset == 'aircraft':
            targets = [s[1] for s in loader.dataset.samples]

        counts = np.unique(targets, return_counts=True)[1]
        class_weights = counts.sum()/(counts*len(counts))
        return torch.Tensor(class_weights)

    class_weights = _get_class_weights(args, loader)

    def custom_acc(logits, labels):
        '''Returns the top1 accuracy, weighted by the class distribution.
        This is important when evaluating an unbalanced dataset. 
        '''
        batch_size = labels.size(0)
        maxk = min(5, logits.shape[-1])
        prec1, _ = tools.helpers.accuracy(
            logits, labels, topk=(1, maxk), exact=True)

        normal_prec1 = prec1.sum(0, keepdim=True).mul_(100/batch_size)
        weighted_prec1 = prec1 * class_weights[labels.cpu()].cuda()
        weighted_prec1 = weighted_prec1.sum(
            0, keepdim=True).mul_(100/batch_size)

        return weighted_prec1.item(), normal_prec1.item()

    return custom_acc


def get_dataset_and_loaders(args):
    '''Given arguments, returns a datasets object and the train and validation loaders.
    '''
    ds, (train_loader, validation_loader) = datasets.make_loaders(
        args.dataset, args.batch_size, 8, subset=None)
    if type(ds) == int:
        new_ds = datasets.CIFAR("/tmp")
        new_ds.num_classes = ds
        new_ds.mean = torch.tensor([0., 0., 0.])
        new_ds.std = torch.tensor([1., 1., 1.])
        ds = new_ds
    return ds, train_loader, validation_loader

    
def replace_fc_layer(model, num_classes):
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model


def load_model_from_id(model_id):
    if model_id == 'megamix':
        return load_megamix_model()
    elif model_id == 'fast_autoaugment':
        return load_fast_autoaugment_model()
    elif model_id == 'mixup':
        return load_mixup_model()
    elif model_id == 'puzzlemix':
        return load_puzzlemix_model()

    elif model_id in constants.MODEL_ID_TO_ROBUSTBENCH_ARGS:
        return load_model_from_robustbench(model_id)

    else:
        raise RuntimeError(f'model_id is not defined for download. Please choose one of {get_supported_model_ids()}')


def get_supported_model_ids():
    return list(constants.MODEL_ID_TO_ROBUSTBENCH_ARGS.keys()) + ['megamix', 'puzzlemix', 'fast_autoaugment', 'mixup']

def load_fast_autoaugment_model(model_path=constants.FAST_AUTOAUGMENT_MODEL_PATH):
    model = resnet50(num_classes=1000)
    
    state_dict_with_module_prepended = torch.load(model_path)
    removed_module_from_state_dict = {k.replace('module.', ''):v for k,v in state_dict_with_module_prepended.items()}
    
    model.load_state_dict(removed_module_from_state_dict)
    return model

def load_mixup_model(model_path=constants.MIXUP_MODEL_PATH):
    model = resnet50(num_classes=1000)
    
    state_dict_with_module_prepended = torch.load(model_path)['state_dict']
    removed_module_from_state_dict = {k.replace('module.', ''):v for k,v in state_dict_with_module_prepended.items()}
    
    model.load_state_dict(removed_module_from_state_dict)
    return model

def load_puzzlemix_model(model_path=constants.PUZZLEMIX_MODEL_PATH):
    model = resnet50(num_classes=1000)
    
    state_dict_with_module_prepended = torch.load(model_path)['state_dict']
    removed_module_from_state_dict = {k.replace('module.', ''):v for k,v in state_dict_with_module_prepended.items()}
    
    model.load_state_dict(removed_module_from_state_dict)
    return model

def load_megamix_model(model_path=constants.MEGAMIX_MODEL_PATH):
    model = resnet50(num_classes=1000)
    
    state_dict_with_module_prepended = torch.load(model_path).state_dict()
    removed_module_from_state_dict = {k.replace('module.', ''):v for k,v in state_dict_with_module_prepended.items()}
    
    model.load_state_dict(removed_module_from_state_dict)
    return model


def load_model_from_robustbench(model_id):
    model = resnet50(num_classes=1000)
    model_args = constants.MODEL_ID_TO_ROBUSTBENCH_ARGS[model_id]
    print(model_args)
    model_from_robustbench = robustbench.load_model(**model_args)
    state_dict_with_model_prepended = model_from_robustbench.state_dict()
    state_dict = {k.replace('model.', ''):v for k,v in state_dict_with_model_prepended.items()}
    for k in ['normalize.mean', 'normalize.std']:
        state_dict.pop(k, 'None')
    model.load_state_dict(state_dict)
    return model


def get_fine_tuning_params(model, freeze_level):
    '''
    Freezes up to args.freeze_level layers of the model (assumes a resnet model)
    '''
    # Freeze layers according to args.freeze-level
    fine_tuning_params = None
    if freeze_level != -1:
        # assumes a resnet architecture
        assert len([name for name, _ in list(model.named_parameters())
                    if f"layer{freeze_level}" in name]), "unknown freeze level (only {1,2,3,4} for ResNets)"
        fine_tuning_params = []
        freeze = True
        for name, param in model.named_parameters():
            print(name, param.size())

            if not freeze and f'layer{freeze_level}' not in name:
                print(f"[Appending the params of {name} to the update list]")
                fine_tuning_params.append(param)
            else:
                param.requires_grad = False

            if freeze and f'layer{freeze_level}' in name:
                # if the freeze level is detected stop freezing onwards
                freeze = False
    return fine_tuning_params