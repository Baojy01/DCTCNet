from .resnet import ResNet18, ResNet34, ResNet50
from .multi_apdctnet import MultiAPSeDCTNet50, MultiAPSeDCTNet34, MultiAPSeDCTNet18
from .multinet import MultiNet50
from .resnext import ResNext50
from .res2net import Res2Net50

models = ['ResNet18', 'ResNet34', 'ResNet50', 'Res2Net50', 'MultiAPSeDCTNet18',
          'MultiAPSeDCTNet34', 'MultiAPSeDCTNet50', 'MultiNet50']


def get_model(name, num_class, kernel_size, ratio):
    if name.lower() == 'resnet18':
        net = ResNet18(num_class)
    elif name.lower() == 'resnet34':
        net = ResNet34(num_class)
    elif name.lower() == 'resnet50':
        net = ResNet50(num_class)
    elif name.lower() == 'resnext50':
        net = ResNext50(num_class)
    elif name.lower() == 'res2net50':
        net = Res2Net50(num_class)
    # ************************************
    elif name.lower() == 'multiapsedctnet18':
        net = MultiAPSeDCTNet18(num_class, kernel_size)
    elif name.lower() == 'multiapsedctnet34':
        net = MultiAPSeDCTNet34(num_class, kernel_size)
    elif name.lower() == 'multiapsedctnet50':
        net = MultiAPSeDCTNet50(num_class, kernel_size)
    elif name.lower() == 'multinet50':
        net = MultiNet50(num_class, kernel_size, ratio)
    else:
        raise NotImplementedError()

    return net
