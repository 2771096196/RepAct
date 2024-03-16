import torch
import GlobalSetting


def modelChoose(args, device):
    import GlobalSetting
    GlobalSetting.act = args.runAct
    print("\nmodel act choose : " + GlobalSetting.act + "\n")

    # mobilenet_v3_small
    if args.runModel == "mobilenet_v3_small":
        import model.mobilenetV3.mobilenetV3
        model = model.mobilenetV3.mobilenetV3.mobilenet_v3_small(
            num_classes=args.num_classes,
        )

    # mobilenet_v3_large
    if args.runModel == "mobilenet_v3_large":
        import model.mobilenetV3.mobilenetV3
        model = model.mobilenetV3.mobilenetV3.mobilenet_v3_large(
            num_classes=args.num_classes,
        )

    # shufflenet_v2_x0_5
    if args.runModel == "shufflenet_v2_x0_5":
        import model.shufflenetV2.shufflenetV2
        model = model.shufflenetV2.shufflenetV2.shufflenet_v2_x0_5(
            num_classes=args.num_classes
        )

    # shufflenet_v2_x1_0
    if args.runModel == "shufflenet_v2_x1_0":
        import model.shufflenetV2.shufflenetV2
        model = model.shufflenetV2.shufflenetV2.shufflenet_v2_x1_0(
            num_classes=args.num_classes
        )

    # swin_tiny_patch4_window7_224
    if args.runModel == "swin_tiny_patch4_window7_224":
        import model.swin_transformer.swinTransformer
        model = model.swin_transformer.swinTransformer.swin_tiny_patch4_window7_224(
            num_classes=args.num_classes
        )

    # vision_transformer
    if args.runModel == "vit_base_patch16_224":
        from model.vision_transformer.vitModel import vit_base_patch16_224
        model = vit_base_patch16_224(
            num_classes=args.num_classes
        )

    if args.runModel == "" or model == None:
        assert "Error"

    torch.autograd.set_detect_anomaly(True)
    model = model.to(device)
    print(model)
    return model
