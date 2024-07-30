from torchvision import datasets, transforms
from torchvision.transforms import autoaugment


def GetData(args):

    if args.dataset.lower() == 'cifar10':
        transform_trains = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_vals = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_sets = datasets.CIFAR10(root="./data/CIFAR10", train=True, download=True, transform=transform_trains)
        val_sets = datasets.CIFAR10(root="./data/CIFAR10", train=False, download=True, transform=transform_vals)
    elif args.dataset.lower() == 'cifar100':
        transform_trains = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_vals = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_sets = datasets.CIFAR100(root="./data/CIFAR100", train=True, download=True, transform=transform_trains)
        val_sets = datasets.CIFAR100(root="./data/CIFAR100", train=False, download=True, transform=transform_vals)

    elif args.dataset.lower() == 'tiny_imagenet':
        transform_trains = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])
        transform_tests = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

        train_sets = datasets.ImageFolder(root="../data/tiny_imagenet/train", transform=transform_trains)
        val_sets = datasets.ImageFolder(root="../data/tiny_imagenet/val", transform=transform_tests)

    elif args.dataset.lower() == 'imagenet20':
        transform_trains = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_vals = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_sets = datasets.ImageFolder(root="../data/ImageNet20/train", transform=transform_trains)
        val_sets = datasets.ImageFolder(root="../data/ImageNet20/val", transform=transform_vals)
    elif args.dataset.lower() == 'imagenet1k':
        transform_trains = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            autoaugment.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_vals = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_sets = datasets.ImageFolder(root=args.data_dir+"/train", transform=transform_trains)
        val_sets = datasets.ImageFolder(root=args.data_dir+"/train", transform=transform_vals)

    else:
        raise NotImplementedError()

    return train_sets, val_sets
