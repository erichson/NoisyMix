from __future__ import print_function
import os
import numpy as np

from robustness import datasets
from robustness.datasets import DataSet, CIFAR
import torch as ch
from constants import DATASET_NAME_TO_PATH, DEFAULT_TRAIN_TRANSFORMS, DEFAULT_TEST_TRANSFORMS

from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader, Subset
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import utils, CIFAR100
from PIL import Image


class ImageNetTransfer(DataSet):
    def __init__(self, data_path, **kwargs):
        ds_kwargs = {
            'num_classes': kwargs['num_classes'],
            'mean': ch.tensor(kwargs['mean']),
            'custom_class': None,
            'std': ch.tensor(kwargs['std']),
            'transform_train': DEFAULT_TRAIN_TRANSFORMS,
            'label_mapping': None,
            'transform_test': DEFAULT_TEST_TRANSFORMS
        }
        super(ImageNetTransfer, self).__init__(kwargs['name'], data_path, **ds_kwargs)

class TransformedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.transform = transform
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample, label = self.ds[idx]
        if self.transform:
            sample = self.transform(sample)
            if sample.shape[0] == 1:
                sample = sample.repeat(3,1,1)
        return sample, label

def make_loaders_pets(batch_size, workers):
    ds = ImageNetTransfer(DATASET_NAME_TO_PATH['pets'], num_classes=37, name='pets',
                            mean=[0., 0., 0.], std=[1., 1., 1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_birds(batch_size, workers):
    ds = ImageNetTransfer(DATASET_NAME_TO_PATH['birds'], num_classes=500, name='birds',
                            mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_SUN(batch_size, workers):
    ds = ImageNetTransfer(DATASET_NAME_TO_PATH['sun397'], num_classes=397, name='SUN397',
                            mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_CIFAR10(batch_size, workers, subset):
    ds = CIFAR('/tmp')
    ds.transform_train = DEFAULT_TRAIN_TRANSFORMS
    ds.transform_test = DEFAULT_TEST_TRANSFORMS
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)

def make_loaders_CIFAR100(batch_size, workers, subset):
    ds = ImageNetTransfer('/tmp', num_classes=100, name='cifar100',
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761])
    ds.custom_class = CIFAR100
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers, subset=subset)

def make_loaders_oxford(batch_size, workers):
    ds = ImageNetTransfer(DATASET_NAME_TO_PATH['flowers'], num_classes=102,
            name='oxford_flowers', mean=[0.,0.,0.],
            std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_aircraft(batch_size, workers):
    ds = ImageNetTransfer(DATASET_NAME_TO_PATH['aircraft'], num_classes=100, name='aircraft',
                    mean=[0.,0.,0.], std=[1.,1.,1.])
    ds.custom_class = FGVCAircraft
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

def make_loaders_food(batch_size, workers):
    food = FOOD101()
    train_ds, valid_ds, classes =  food.get_dataset()
    train_dl, valid_dl = food.get_dls(train_ds, valid_ds, bs=batch_size,
                                                    num_workers=workers)
    return 101, (train_dl, valid_dl)

def make_loaders_caltech101(batch_size, workers):
    ds = Caltech101(DATASET_NAME_TO_PATH['caltech101'], download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 30

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    train_set = TransformedDataset(train_set, transform=DEFAULT_TRAIN_TRANSFORMS) 
    test_set = TransformedDataset(test_set, transform=DEFAULT_TEST_TRANSFORMS)

    return 101, [DataLoader(d, batch_size=batch_size, shuffle=True,
                num_workers=workers) for d in (train_set, test_set)]

def make_loaders_caltech256(batch_size, workers):
    ds = Caltech256(DATASET_NAME_TO_PATH['caltech256'], download=True)
    np.random.seed(0)
    ch.manual_seed(0)
    ch.cuda.manual_seed(0)
    ch.cuda.manual_seed_all(0)
    NUM_TRAINING_SAMPLES_PER_CLASS = 60

    class_start_idx = [0]+ [i for i in np.arange(1, len(ds)) if ds.y[i]==ds.y[i-1]+1]

    train_indices = sum([np.arange(start_idx,start_idx + NUM_TRAINING_SAMPLES_PER_CLASS).tolist() for start_idx in class_start_idx],[])
    test_indices = list((set(np.arange(1, len(ds))) - set(train_indices) ))

    train_set = Subset(ds, train_indices)
    test_set = Subset(ds, test_indices)

    train_set = TransformedDataset(train_set, transform=DEFAULT_TRAIN_TRANSFORMS) 
    test_set = TransformedDataset(test_set, transform=DEFAULT_TEST_TRANSFORMS)

    return 257, [DataLoader(d, batch_size=batch_size, shuffle=True,
                num_workers=workers) for d in (train_set, test_set)]

def make_loaders_dtd(batch_size, workers):
        train_set = DTD(train=True)
        val_set = DTD(train=False)
        return 57, [DataLoader(ds, batch_size=batch_size, shuffle=True,
                num_workers=workers) for ds in (train_set, val_set)]

def make_loaders_cars(batch_size, workers):
    ds = ImageNetTransfer(DATASET_NAME_TO_PATH['cars'], num_classes=196, name='stanford_cars',
                            mean=[0.,0.,0.], std=[1.,1.,1.])
    return ds, ds.make_loaders(batch_size=batch_size, workers=workers)

DS_TO_FUNC = {
    "dtd": make_loaders_dtd,
    "stanford_cars": make_loaders_cars,
    "cifar10": make_loaders_CIFAR10,
    "cifar100": make_loaders_CIFAR100,
    "sun397": make_loaders_SUN,
    "aircraft": make_loaders_aircraft,
    "flowers": make_loaders_oxford,
    "food": make_loaders_food,
    "birds": make_loaders_birds,
    "caltech101": make_loaders_caltech101,
    "caltech256": make_loaders_caltech256,
    "pets": make_loaders_pets,
}

def make_loaders(ds, batch_size, workers, subset):
    if ds in ['cifar10', 'cifar100']:
        return DS_TO_FUNC[ds](batch_size, workers, subset)
    
    if subset: raise Exception(f'Subset not supported for the {ds} dataset')
    return DS_TO_FUNC[ds](batch_size, workers)


def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


class FGVCAircraft(data.Dataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.
    Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root, class_type='variant', train=True, transform=None,
                 target_transform=None, loader=default_loader, download=False):
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))
        self.root = os.path.expanduser(root)
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data', 'images')) and \
            os.path.exists(self.classes_file)

    def download(self):
        """Download the FGVC-Aircraft data if it doesn't exist already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s ... (may take a few minutes)' % self.url)
        parent_dir = os.path.abspath(os.path.join(self.root, os.pardir))
        tar_name = self.url.rpartition('/')[-1]
        tar_path = os.path.join(parent_dir, tar_name)
        data = urllib.request.urlopen(self.url)

        # download .tar.gz file
        with open(tar_path, 'wb') as f:
            f.write(data.read())

        # extract .tar.gz to PARENT_DIR/fgvc-aircraft-2013b
        data_folder = tar_path.strip('.tar.gz')
        print('Extracting %s to %s ... (may take a few minutes)' % (tar_path, data_folder))
        tar = tarfile.open(tar_path)
        tar.extractall(parent_dir)

        # if necessary, rename data folder to self.root
        if not os.path.samefile(data_folder, self.root):
            print('Renaming %s to %s ...' % (data_folder, self.root))
            os.rename(data_folder, self.root)

        # delete .tar.gz file
        print('Deleting %s ...' % tar_path)
        os.remove(tar_path)

        print('Done!')


class FOOD101():
    def __init__(self):
        self.TRAIN_PATH = DATASET_NAME_TO_PATH['food']+"/train"
        self.VALID_PATH = DATASET_NAME_TO_PATH['food']+"/valid"

        self.train_ds, self.valid_ds, self.train_cls, self.valid_cls = [None]*4
   
    def _get_tfms(self):
        train_tfms = DEFAULT_TRAIN_TRANSFORMS
        valid_tfms = DEFAULT_TEST_TRANSFORMS
        return train_tfms, valid_tfms            
            
    def get_dataset(self):
        train_tfms, valid_tfms = self._get_tfms() # transformations
        self.train_ds = datasets.ImageFolder(root=self.TRAIN_PATH,
                                        transform=train_tfms)
        self.valid_ds = datasets.ImageFolder(root=self.VALID_PATH,
                                        transform=valid_tfms)        
        self.train_classes = self.train_ds.classes
        self.valid_classes = self.valid_ds.classes

        assert self.train_classes==self.valid_classes
        return self.train_ds, self.valid_ds, self.train_classes
    
    def get_dls(self, train_ds, valid_ds, bs, **kwargs):
        return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
               DataLoader(valid_ds, batch_size=bs, shuffle=True, **kwargs))


class DTD(Dataset):
    def __init__(self, split="1", train=False):
        super().__init__()
        train_path = os.path.join(DATASET_NAME_TO_PATH['dtd'], f"labels/train{split}.txt")
        val_path = os.path.join(DATASET_NAME_TO_PATH['dtd'], f"labels/val{split}.txt")
        test_path = os.path.join(DATASET_NAME_TO_PATH['dtd'], f"labels/test{split}.txt")
        if train:
            self.ims = open(train_path).readlines() + \
                            open(val_path).readlines()
        else:
            self.ims = open(test_path).readlines()
        
        self.full_ims = [os.path.join(DATASET_NAME_TO_PATH['dtd'], "images", x) for x in self.ims]
        
        pth = os.path.join(DATASET_NAME_TO_PATH['dtd'], f"labels/classes.txt")
        self.c_to_t = {x.strip(): i for i, x in enumerate(open(pth).readlines())}

        self.transform = DEFAULT_TRAIN_TRANSFORMS if train else \
                                         DEFAULT_TEST_TRANSFORMS
        self.labels = [self.c_to_t[x.split("/")[0]] for x in self.ims]

    def __getitem__(self, index):
        im = Image.open(self.full_ims[index].strip())
        im = self.transform(im)
        return im, self.labels[index]

    def __len__(self):
        return len(self.ims)


class Caltech101(VisionDataset):
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.
    .. warning::
        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.
    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
        ``annotation``. Can also be a list to output a tuple with all specified target types.
        ``category`` represents the target class, and ``annotation`` is a list of points
        from a hand-generated outline. Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, target_type="category", transform=None,
                 target_transform=None, download=False):
        super(Caltech101, self).__init__(os.path.join(root, 'caltech101'),
                                         transform=transform,
                                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = [utils.verify_str_arg(t, "target_type", ("category", "annotation"))
                            for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {"Faces": "Faces_2",
                    "Faces_easy": "Faces_3",
                    "Motorbikes": "Motorbikes_16",
                    "airplanes": "Airplanes_Side_2"}
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(os.path.join(self.root,
                                      "101_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "image_{:04d}.jpg".format(self.index[index])))

        target = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(os.path.join(self.root,
                                                     "Annotations",
                                                     self.annotation_categories[self.y[index]],
                                                     "annotation_{:04d}.mat".format(self.index[index])))
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self):
        return len(self.index)

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        utils.download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9")
        utils.download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
            self.root,
            filename="101_Annotations.tar",
            md5="6f83eeb1f24d99cab4eb377263132c91")

    def extra_repr(self):
        return "Target type: {target_type}".format(**self.__dict__)


class Caltech256(VisionDataset):
    """`Caltech 256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, transform=None, target_transform=None, download=False):
        super(Caltech256, self).__init__(os.path.join(root, 'caltech256'),
                                         transform=transform,
                                         target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            image_paths = os.listdir(os.path.join(self.root, "256_ObjectCategories", c))
            image_paths = [path for path in image_paths if '.jpg' in path]
            n = len(image_paths)
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(os.path.join(self.root,
                                      "256_ObjectCategories",
                                      self.categories[self.y[index]],
                                      "{:03d}_{:04d}.jpg".format(self.y[index] + 1, self.index[index])))

        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def __len__(self):
        return len(self.index)

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        utils.download_and_extract_archive(
            "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
            self.root,
            filename="256_ObjectCategories.tar",
            md5="67b4f42ca05d46448c6bb8ecd2220f6d")