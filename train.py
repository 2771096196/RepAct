import os
import math
import argparse
import random
import warnings

import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from train_utils.utilsTrain import train_one_epoch, evaluate, CrossEntropyLabelSmooth

best_acc1 = 0


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    global best_acc1
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    if args.autoDL:
        path_Pre = '/root/tf-logs/runs/'
    else:
        path_Pre = './runs/'
    args.runTitle = args.runModel + "_" + args.runAct
    SavePath = path_Pre + args.runTitle + "____" + args.datasets \
               + "_" + str(args.titleSupplement) + \
               "_" + str(args.L2) + "_" + str(args.lr) + "_" + str(args.batch_size)
    tb_writer = SummaryWriter(SavePath)
    with open(SavePath + "/" + "saveArgs.txt", 'w') as f:
        import json
        args_dict = vars(args)
        json.dump(args_dict, f)
    from train_utils.ReadOwnData import create_data_loaders_train_val
    train_loader, val_loader = create_data_loaders_train_val(args)

    # model
    from model.modelChoose import modelChoose
    model = modelChoose(args, device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=float(args.L2))
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    criterion_smooth = CrossEntropyLabelSmooth(args.num_classes, args.CrossEntropyLabelSmooth)
    LossFunc = criterion_smooth.to(device)

    args.start_epoch = False
    # 如果存在预训练权重则载入
    if args.weights != "":
        if os.path.exists(args.weights):
            checkpoint = torch.load(args.weights, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loading Pretrain Model ", args.weights)
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))
    for i in range(args.start_epoch):
        scheduler.step()
    from RepActs.visionRepActTrain import plot_repact_layers
    plot_repact_layers(model, SavePath + "/" + str(-1), str(-1))
    for epoch in range(args.start_epoch, args.epochs) if args.start_epoch else range(args.epochs):
        # 记录模型权重分布
        for name, param in model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            tb_writer.add_histogram("weightDist/{}/{}".format(layer, attr), param, epoch)
        # train
        # Train_loss, Train_acc1, Train_acc5 = 0,0,0
        Train_loss, Train_acc1, Train_acc5 = \
            train_one_epoch(model=model,
                            optimizer=optimizer,
                            data_loader=train_loader,
                            device=device,
                            epoch=epoch,
                            loss_function=LossFunc,
                            args=args)
        scheduler.step()
        # validate
        # Val_loss, Val_acc1, Val_acc5 = 0,0,0
        Val_loss, Val_acc1, Val_acc5 = evaluate(model=model,
                                                data_loader=val_loader,
                                                device=device,
                                                epoch=epoch,
                                                loss_function=LossFunc,
                                                args=args)
        # tensorboard
        tb_writer.add_scalar("train_loss", Train_loss, epoch)
        tb_writer.add_scalar("train_acc1", Train_acc1, epoch)
        tb_writer.add_scalar("train_acc5", Train_acc5, epoch)
        tb_writer.add_scalar("val_loss", Val_loss, epoch)
        tb_writer.add_scalar("val_acc1", Val_acc1, epoch)
        tb_writer.add_scalar("val_acc5", Val_acc5, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # remember best val_acc1 and save checkpoint
        is_best = Val_acc1 > best_acc1
        best_acc1 = max(Val_acc1, best_acc1)
        print('best val Top-1 Acc:', best_acc1)
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }
        # Save Last every epoch
        torch.save(state, SavePath + "/LastModel.pth")
        # Save Best
        if is_best:
            torch.save(state, SavePath + "/BestModel.pth")
        # plot RepAct
        plot_repact_layers(model, SavePath + "/" + str(epoch), str(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runModel',
                        default='mobilenet_v3_small',
                        help='mobilenet_v3_small'
                             'mobilenet_v3_large'
                             'shufflenet_v2_x0_5'
                             'shufflenet_v2_x1_0'
                             'swin_tiny_patch4_window7_224'
                             'vit_base_patch16_224'
                        )
    parser.add_argument('--runAct',
                        default='RepAct_Softmax_Reconstruct',
                        help='See ActLists in file ./model/actFactory.py ')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.0001)
    parser.add_argument('--L2', default=0.0004,
                        help='L2')
    parser.add_argument('--CrossEntropyLabelSmooth', default=0.1,
                        help='0.1')
    parser.add_argument('--titleSupplement', default='',
                        help='titleSupplement')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--datasets', default='self_train_val',
                        help='self_train_val')
    parser.add_argument('--data-path', type=str,
                        default=r"./ImageNet100/train")
    parser.add_argument('--data-val-path', type=str,
                        default=r"./ImageNet100/val")
    parser.add_argument('--autoDL', default=False,
                        help='is use autoDL')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--print_freq_step', type=int, default='10', help='print info of Train or Val')
    parser.add_argument('--grad_save_n_iter', type=int, default='100', help='print info of Train or Val')
    parser.add_argument('--seed', type=int, default='0', help='seed')
    opt = parser.parse_args()
    main(opt)
