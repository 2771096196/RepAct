import os
import argparse
import torch
from train_utils.ReadOwnData import create_data_loaders_split, create_data_loaders_train_val
from train_utils.utilsTrain import evaluate, CrossEntropyLabelSmooth
from RepActs.validRep.RepAct_Origin_Reconstruct import RepAct_Origin_Reconstruct
from RepActs.validRep.RepAct_BN_Reconstruct import RepAct_BN_Reconstruct
from RepActs.validRep.RepAct_Softmax_Reconstruct import RepAct_Softmax_Reconstruct
best_acc1 = 0
def traverse_model(model):
    model_list = list(model.modules())
    for module in model_list:
        if isinstance(module, RepAct_Origin_Reconstruct):
            module.inference = True
            module.RepActFuse()
            print("RepAct_Origin_Reconstruct Fused")
        if isinstance(module, RepAct_Softmax_Reconstruct):
            module.inference = True
            module.RepActFuse()
            print("RepAct_Softmax_Reconstruct Fused")
        if isinstance(module, RepAct_BN_Reconstruct):
            module.inference = True
            module.RepActFuse()
            print("RepAct_BN_Reconstruct Fused")

def validate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    global best_acc1
    print(args)
    if args.datasets == "self_train_val":
        _, val_loader = create_data_loaders_train_val(args)
    else:
        print("error with datasets choose")
    # Model creation (similar to the training script)
    model = create_model(args)
    model = model.to(device)
    # Load the trained weights
    if args.weights != "":
        if os.path.exists(args.weights):
            checkpoint = torch.load(args.weights, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print("Loading Pretrained Model ", args.weights)
        else:
            raise FileNotFoundError("Weights file not found: {}".format(args.weights))
    model.eval()
    if (args.FuseModel):
        print("FuseModel:", args.FuseModel)
        traverse_model(model)
    # Loss function and evaluation criterion
    criterion_smooth = CrossEntropyLabelSmooth(args.num_classes, args.CrossEntropyLabelSmooth)
    LossFunc = criterion_smooth.to(device)
    # Evaluate the model on the validation set
    Val_loss, Val_acc1, Val_acc5 = evaluate(model=model,
                                            data_loader=val_loader,
                                            device=device,
                                            epoch=0,
                                            loss_function=LossFunc,
                                            args=args)
    print('Validation Loss:', Val_loss)
    print('Validation Top-1 Accuracy:', Val_acc1)
    print('Validation Top-5 Accuracy:', Val_acc5)

def create_model(args):
    # Model creation based on runTitle (similar to the training script)
    # Add the necessary imports for model creation
    from model.modelChoose import modelChoose
    model = modelChoose(args, args.device)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--FuseModel', default=True, help='True means to valid RepAct_xxx_Reconstruct')
    parser.add_argument('--autoDL', default=False, help='is use autoDL')
    parser.add_argument('--print_freq_step', default=10, help='')
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
    parser.add_argument('--weights', type=str,
                        default=r'./runs/mobilenet_v3_small_RepAct_Softmax_Reconstruct____self_train_val__0.0004_0.01_32/BestModel.pth',
                        help='path to trained weights')
    parser.add_argument('--data-path', type=str,
                        default=r"D:\ImageNet100\train")
    parser.add_argument('--data-val-path',
                        default=r"D:\ImageNet100\val")
    parser.add_argument('--datasets', default='self_train_val', help='dataset choice')  # Change here
    parser.add_argument('--titleSupplement', default='', help='titleSupplement')
    parser.add_argument('--CrossEntropyLabelSmooth', default=0.1, help='label smoothing strength')
    parser.add_argument('--num_classes', type=int, default=100, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e., cuda:0 or cpu)')
    opt = parser.parse_args()
    validate(opt)
