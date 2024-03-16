import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from model.mobilenetV3.mobilenetV3 import mobilenet_v3_small
from utils import GradCAM, show_cam_on_image, center_crop_img


def process_image_with_gradcam(modelName, model, model_weight_path, target_layers, img_folder, target_category):
    # Load pretrain weights
    assert os.path.exists(model_weight_path), "Cannot find {} file".format(model_weight_path)
    temp = torch.load(model_weight_path)
    model.load_state_dict(temp['state_dict'])

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Create output folder based on the model name
    model_name = os.path.basename(model_weight_path).replace(".pth", "")
    output_folder = os.path.join(f"./camOut/{modelName}——LastConv——{target_category}")
    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(img_folder):
        # Load image
        img_path = os.path.join(img_folder, img_file)
        assert os.path.exists(img_path), "File: '{}' does not exist.".format(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)

        img_tensor = data_transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., grayscale_cam, use_rgb=True)

        # Save the image
        output_path = os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_gradcam.jpg")
        plt.imsave(output_path, visualization)
    # 缩略图工作
    # 目标大小
    target_size = (100, 100)
    # 每行图片数量
    images_per_row = 10
    input_folder_path = output_folder
    output_file_path = output_folder + "/concat.png"
    from tools.grad_cam.pigcat import resize_and_concatenate_images
    resize_and_concatenate_images(input_folder_path, output_file_path, target_size, images_per_row)

    return output_folder


if __name__ == '__main__':
    img_folders_target_categories = [
        (r"D:\ImageNet100\val\n01443537", 1),
        (r"D:\ImageNet100\val\n02002556", 86),
        (r"D:\ImageNet100\val\n01773797", 50),
        (r"D:\ImageNet100\val\n02051845", 97),
        # (r"D:\ImageNet100\val\n01677366", 27),
        # (r"D:\ImageNet100\val\n01756291", 45),
        # (r"D:\ImageNet100\val\n01843383", 66),
        # (r"D:\ImageNet100\val\n01955084", 79),
    ]

    for i in range(len(img_folders_target_categories)):
        img_folder, target_category = img_folders_target_categories[i]
        target_size = (100, 100)
        images_per_row = 10
        input_folder_path = img_folder
        output_file_path = "camOut/class/" + str(target_category) + "_concat.png"
        from tools.grad_cam.pigcat import resize_and_concatenate_images

        resize_and_concatenate_images(input_folder_path, output_file_path, target_size, images_per_row)

        # mobilenet_v3_small_RepAct_Softmax
        import GlobalSetting
        GlobalSetting.act = "RepAct_Softmax"
        print("\nmodel act choose : " + GlobalSetting.act + "\n")
        modelName = "mobilenet_v3_small_RepAct_Softmax"
        model = mobilenet_v3_small(num_classes=100)
        model_weight_path = r"./runs/mobilenet_v3_small_RepAct_Softmax____self_train_val__0.0004_0.01_32/BestModel.pth"
        target_layers = [model.LastConv]
        result_folder = process_image_with_gradcam(modelName, model, model_weight_path, target_layers, img_folder,
                                                   target_category)

