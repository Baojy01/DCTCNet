import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.backends.cudnn

from models import config
from models import get_model
from utils import GetData, train_runner, val_runner

model_names = config.models
parser = argparse.ArgumentParser(description='PyTorch image training')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training')
parser.add_argument('--device', default='cuda:0', type=str, help='device')
parser.add_argument('--dataset', default='tiny_imagenet', type=str, choices=['imagenet1k', 'cifar10', 'cifar100'], help='dataset for training')
parser.add_argument('--num_classes', default=200, type=int, help='number of classes for classification')
parser.add_argument('--data_dir', default='', type=str, help='path of dataset')
parser.add_argument('--arch', default='MobileNetV2', type=str, choices=model_names, metavar='ARCH', help='model architecture')
parser.add_argument('--use_dct', default=(False, False, True), help='whether using apdct or not')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='EP', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='SE', help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup', default=True, type=bool, help='whether using warmup or not  (default: True)')
parser.add_argument('--warmup_epoch', default=5, type=int, metavar='WE', help='use warmup epoch number (default: 5)')
parser.add_argument('--kernel_size', default=3, type=int, metavar='KS', help='APSeDCT kernel size, it must be an odd and not less than 3')
parser.add_argument('--batch_size', default=128, type=int, metavar='BS', help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default=None, type=str, help='directory if using pre-trained model')
parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def main():
    if args.seed is not None:
        set_seed(args.seed)

    train_set, val_set = GetData(args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device))

    model = get_model(args)
    model = model.to(device)

    dct_flag = ''.join([str(1) if x else str(0) for x in args.use_dct])

    results_dir = './results/%s/' % (args.arch + '_' + args.dataset + '_bs_' + str(args.batch_size) + '_k_' + str(args.kernel_size)
                                     + '_epochs_' + str(args.epochs) + '_' + dct_flag)
    save_dir = './checkpoint/%s/' % (args.arch + '_' + args.dataset + '_bs_' + str(args.batch_size) + '_k_' + str(args.kernel_size)
                                     + '_epochs_' + str(args.epochs) + '_' + dct_flag)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    log_dir = os.path.join(save_dir, "last_model.pth")
    train_dir = os.path.join(results_dir, "train.csv")
    val_dir = os.path.join(results_dir, "val.csv")

    best_top1, best_top5 = 0.0, 0.0
    Loss_train, Accuracy_train_top1, Accuracy_train_top5 = [], [], []
    Loss_val, Accuracy_val_top1, Accuracy_val_top5 = [], [], []

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
        elif os.path.isfile(log_dir):
            print("There is no checkpoint found at '{}', then loading default checkpoint at '{}'.".format(args.resume, log_dir))
            checkpoint = torch.load(log_dir)  # default load last_model.pth
        else:
            raise FileNotFoundError()

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

        if args.start_epoch < args.epochs:
            best_top1, best_top5 = checkpoint['best_top1'], checkpoint['best_top5']
            print('Loading model successfully, current start_epoch={}.'.format(args.start_epoch))
            trainF = open(train_dir, 'a+')
            valF = open(val_dir, 'a+')
        else:
            raise ValueError('epochs={}, but start_epoch={} in the saved model, please reset epochs larger!'.format(args.epochs, args.start_epoch))
    else:
        trainF = open(results_dir + 'train.csv', 'w')
        valF = open(results_dir + 'val.csv', 'w')
        trainF.write('{},{},{},{}\n'.format('epoch', 'loss', 'top1', 'top5'))
        valF.write('{},{},{},{},{},{}\n'.format('epoch', 'val_loss', 'val_top1', 'val_top5', 'best_top1', 'best_top5'))

    for epoch in range(args.start_epoch, args.epochs):

        time_star = time.time()
        top1, top5, loss = train_runner(model, device, train_loader, criterion, optimizer, lr_scheduler, args, epoch, scaler=scaler)
        val_top1, val_top5, val_loss = val_runner(model, device, val_loader)
        time_end = time.time()

        lr = optimizer.param_groups[0]["lr"]

        if args.warmup:
            if epoch >= args.warmup_epoch:
                lr_scheduler.step()
        else:
            lr_scheduler.step()

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'best_top1': best_top1,
            'best_top5': best_top5,
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()

        if best_top5 < val_top5:
            best_top5 = val_top5

        if best_top1 < val_top1:
            best_top1 = val_top1
            torch.save(save_files, os.path.join(save_dir, "best_model.pth"))

        torch.save(save_files, log_dir)

        Loss_train.append(loss)
        Accuracy_train_top1.append(top1)
        Accuracy_train_top5.append(top5)

        Loss_val.append(val_loss)
        Accuracy_val_top1.append(val_top1)
        Accuracy_val_top5.append(val_top5)

        print("Train Epoch: {} \t train loss: {:.4f}, train top1: {:.4f}%, train top5: {:.4f}%".format(epoch, loss, 100.0 * top1, 100.0 * top5))
        print("val_loss: {:.4f}, val_top1: {:.4f}%, val_top5: {:.4f}%".format(val_loss, 100.0 * val_top1, 100.0 * val_top5))
        print("best_val_top1: {:.4f}%, best_val_top5: {:.4f}%, lr: {:.6f}".format(100.0 * best_top1, 100.0 * best_top5, lr))
        print('Each epoch running time: {:.4f} s'.format(time_end - time_star))
        trainF.write('{},{},{},{},{}\n'.format(epoch, loss, top1, top5, lr))
        valF.write('{},{},{},{},{},{}\n'.format(epoch, val_loss, val_top1, val_top5, best_top1, best_top5))

        trainF.flush()
        valF.flush()

    trainF.close()
    valF.close()

    print('Finished Training!')


if __name__ == "__main__":
    main()
