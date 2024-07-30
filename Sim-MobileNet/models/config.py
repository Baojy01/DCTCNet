from .mobilenet_v2 import MobileNetV2

models = ['MobileNetV2']


def get_model(args):
    if args.arch.lower() == 'mobilenetv2':
        net = MobileNetV2(args.num_classes, args.use_dct)
    else:
        raise NotImplementedError()

    return net
