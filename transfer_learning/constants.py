MEGAMIX_MODEL_PATH = 'imagenet_models/'\
                'final_arch_resnet50_augmix_1_jsd_1_alpha_1.0_manimixup_1_addn_0.4_multn_0.4_seed_1.pt'

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
