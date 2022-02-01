
NOISYMIX_MODEL_PATH = 'Insert path gere'

PUZZLEMIX_MODEL_PATH = 'Insert path gere'
FAST_AUTOAUGMENT_MODEL_PATH = 'Insert path gere'
MIXUP_MODEL_PATH = 'Insert path gere'


MODEL_ID_TO_ROBUSTBENCH_ARGS = {
    'augmix':{'model_name' : 'Hendrycks2020AugMix', 
            'dataset' : 'imagenet', 
            'threat_model' : 'corruptions'},

    'adversarial':{'model_name' : 'Salman2020Do_R50', 
                   'dataset' : 'imagenet', 
                   'threat_model' : 'Linf'},

    'standard':{'model_name' : 'Standard_R50', 
                'dataset' : 'imagenet', 
                'threat_model' : 'Linf'},
}

DATASET_NAME_TO_PATH = {
    'aircraft': 'downloaded_datasets/fgvc-aircraft-2013b/',
    'flowers': 'downloaded_datasets/flowers_new/',
    'dtd': 'downloaded_datasets/dtd/',
    'cars': 'downloaded_datasets/cars_new',
    'sun397':'downloaded_datasets/SUN397/splits_01/',
    'food': 'downloaded_datasets/food-101',
    'birds': 'downloaded_datasets/birdsnap',
    'pets': 'downloaded_datasets/pets',
    'caltech101': 'downloaded_datasets/caltech101',
    'caltech256': 'downloaded_datasets/caltech256',
    'cifar10': '/tmp',
    'cifar100': '/tmp'
}

from torchvision import transforms
# Data Augmentation defaults
DEFAULT_TRAIN_TRANSFORMS = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

DEFAULT_TEST_TRANSFORMS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
