MEGAMIX_MODEL_PATH = '/home/ubuntu/MegaMix/imagenet_models/'\
                'best_arch_resnet50_augmix_1_jsd_1_alpha_1.0_manimixup_1_addn_0.4_multn_0.4_seed_1.pt'

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
    'planes': '/tmp/datasets/fgvc-aircraft-2013b/',
    'flowers': '/tmp/datasets/oxford_flowers_pytorch/',
    'dtd': '/tmp/datasets/dtd/',
    'cars': '/tmp/datasets/cars_new',
    'sun397':'/tmp/datasets/SUN397/splits_01/',
    'food': '/tmp/datasets/food-101',
    'birds': '/tmp/datasets/birdsnap',
    'pets': '/tmp/datasets/pets',
    'caltech101': '/tmp/datasets',
    'caltech256': '/tmp/datasets',
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