
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.insert(0, './')
print(sys.path)
from classifier import ImageClassifier as Classifier
from utils.data import ForeverDataIterator
from utils.metric import accuracy
from utils.meter import AverageMeter, ProgressMeter
from utils.logger import CompleteLogger
from utils.analysis import tsne, a_distance
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def main(args: argparse.Namespace):

    log_name = args.log + args.data + '_' + args.arch + '_src_' + '_'.join(args.sources)

    logger = CompleteLogger(log_name, args.phase)
    logger.write(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    print(args)

    if args.data == 'DomainNet':
        test_iter = 4
    else:
        test_iter = 2

    logger.write('gpu count: {}'.format(torch.cuda.device_count()))
    if torch.cuda.device_count() >= 1:
        logger.write('gpu name: {}'.format(torch.cuda.get_device_name(0)))

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=True,
                                                random_color_jitter=True, random_gray_scale=True)
    val_transform = utils.get_val_transform(args.val_resizing)
    logger.write("train_transform: {}".format(train_transform))
    logger.write("val_transform: {}".format(val_transform))

    train_dataset, num_classes = utils.get_dataset(dataset_name=args.data, root=args.root, task_list=args.sources,
                                                   split='train', download=True, transform=train_transform,
                                                   seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset, _ = utils.get_dataset(dataset_name=args.data, root=args.root, task_list=args.sources, split='val',
                                       download=True, transform=val_transform, seed=args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_dataset, _ = utils.get_dataset(dataset_name=args.data, root=args.root, task_list=args.targets, split='test',
                                        download=True, transform=val_transform, seed=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    logger.write("train_dataset_size: {}".format(len(train_dataset)))
    logger.write('val_dataset_size: {}'.format(len(val_dataset)))
    logger.write("test_dataset_size: {}".format(len(test_dataset)))
    train_iter = ForeverDataIterator(train_loader)

    # create model
    logger.write("=> using pre-trained model '{}'".format(args.arch))

    test_val_acc1 = 0.
    global_best_val_acc1 = 0.

    for opt in ['sgd', 'adam']:

        if opt == 'sgd':
            lr_list  = [1e-2, 1e-3, 1e-3]
        if opt == 'adam':
            lr_list  = [1e-3, 1e-4, 1e-5]

        for lr in lr_list:
            args.lr = lr

            for seed in [0]:
                args.seed = seed
                if args.seed is not None:
                    random.seed(args.seed)
                    torch.manual_seed(args.seed)

                logger.write('opt:{} lr:{} seed:{}'.format(opt, lr, seed))
                model_name = args.arch

                if model_name == 'dino_vits16':
                    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
                    backbone.out_features = 384
                elif model_name == 'dino_vitb16':
                    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
                    backbone.out_features = 768
                elif model_name == 'dino_resnet50':
                    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
                    backbone.out_features = 2048

                else:
                    backbone = utils.get_model(args.arch)


                pool_layer = nn.Identity() if args.no_pool else None
                classifier = Classifier(backbone, num_classes, freeze_bn=args.freeze_bn, dropout_p=args.dropout_p,
                                        finetune=args.finetune, pool_layer=pool_layer).to(device)

                # define optimizer and lr scheduler
                if opt == 'sgd':
                    optimizer = SGD(classifier.get_parameters(base_lr=args.lr), args.lr, momentum=args.momentum, weight_decay=args.wd,
                                nesterov=True)
                else:
                    optimizer = Adam(classifier.get_parameters(base_lr=args.lr), args.lr)

                lr_scheduler = CosineAnnealingLR(optimizer, args.epochs * args.iters_per_epoch)

                # resume from the best checkpoint
                if args.phase != 'train':
                    checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
                    classifier.load_state_dict(checkpoint)


                # analysis the model
                if args.phase == 'analysis':
                    # extract features from both domains
                    feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
                    source_feature = utils.collect_feature(val_loader, feature_extractor, device, max_num_features=100)
                    target_feature = utils.collect_feature(test_loader, feature_extractor, device, max_num_features=100)
                    print(len(source_feature), len(target_feature))
                    # plot t-SNE
                    tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
                    tsne.visualize(source_feature, target_feature, tSNE_filename)
                    logger.write("Saving t-SNE to", tSNE_filename)
                    # calculate A-distance, which is a measure for distribution discrepancy
                    A_distance = a_distance.calculate(source_feature, target_feature, device)
                    logger.write("A-distance =", A_distance)
                    return

                if args.phase == 'test':
                    acc1 = utils.validate_each_domain(test_loader, classifier, args, device, logger)
                    return


                best_val_acc1 = 0.
                best_test_acc1 = 0.
                for epoch in range(args.epochs):
                    print(lr_scheduler.get_lr())
                    # train for one epoch
                    train(train_iter, classifier, optimizer, lr_scheduler, epoch, args)

                    # evaluate on validation set
                    print("Evaluate on validation set...")

                    if epoch % test_iter == 1:
                        acc1 = utils.validate(val_loader, classifier, args, device)
                        acc1_list, test_acc = utils.validate_domains(test_loader, classifier, args, device, logger)

                        logger.write("Evaluate on test set...")
                        if acc1 > best_val_acc1:
                            best_val_acc1 = max(acc1, best_val_acc1)


                        if acc1 > global_best_val_acc1:
                            global_best_val_acc1 = acc1
                            test_val_acc1 = test_acc
                            test_val_acc1_list = acc1_list
                            torch.save({"state_dict": classifier.state_dict(), "lr":args.lr, "opt":opt}, logger.get_checkpoint_path('global_best'))

                            best_test_acc1 = max(best_test_acc1, test_acc)



    max_acc = test_val_acc1

    logger.write("{}".format(args.arch))
    logger.write('{:.2f}'.format(max_acc))

    str_list = ""
    for target, acc in zip(args.targets, test_val_acc1_list):
        str_list += "{}: {} ".format(target, '{:.2f}'.format(acc))

    logger.write('{:.2f}'.format(str_list))
    logger.write('Source: {} acc: {:.2f}'.format(args.sources, global_best_val_acc1))

    logger.close()



def train(train_iter: ForeverDataIterator, model: Classifier, optimizer,
          lr_scheduler: CosineAnnealingLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x, labels, _ = next(train_iter)
        x = x.to(device)
        labels = labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y, _ = model(x)

        loss = F.cross_entropy(y, labels)

        cls_acc = accuracy(y, labels)[0]
        losses.update(loss.item(), x.size(0))
        cls_accs.update(cls_acc.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline for Domain Generalization')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='PACS')
                        # help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                        #      ' (default: PACS)')
    parser.add_argument('-s', '--sources', nargs='+', default=None,
                        help='source domain(s)')
    parser.add_argument('-t', '--targets', nargs='+', default=None,
                        help='target domain(s)')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='deit_small_patch16_')
                        # choices=utils.get_model_names(),
                        # help='backbone architecture: ' +
                        #      ' | '.join(utils.get_model_names()) +
                        #      ' (default: resnet50)')
    # deit_small_patch16_224
    parser.add_argument('--no-pool', action='store_true', help='no pool layer after the feature extractor.')
    parser.add_argument('--finetune', default=True, action='store_true', help='whether use 10x smaller lr for backbone')
    parser.add_argument('--freeze-bn', action='store_true', help='whether freeze all bn layers')
    parser.add_argument('--dropout-p', type=float, default=0.1, help='only activated when freeze-bn is True')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=250, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
