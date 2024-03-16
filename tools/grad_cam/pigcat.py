from PIL import Image
import os

from PIL import Image
import os

def resize_and_concatenate_images(input_folder, output_path, target_size=(224, 224), images_per_row=10):
    # 获取输入文件夹中所有图片文件的列表
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]

    # 调整每张图片的大小
    resized_images = []
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        image = Image.open(input_path)
        image = image.resize(target_size, Image.ANTIALIAS)
        resized_images.append(image)

    # 计算输出图片的大小
    image_width, image_height = target_size[0] * images_per_row, target_size[1] * ((len(resized_images) - 1) // images_per_row + 1)

    # 创建新的大图
    result_image = Image.new("RGB", (image_width, image_height), (255, 255, 255))

    # 将调整大小后的图片拼接到大图上
    for i, image in enumerate(resized_images):
        row = i // images_per_row
        col = i % images_per_row
        x = col * target_size[0]
        y = row * target_size[1]

        # Ensure the calculated position does not exceed the size of the result image
        if x < image_width and y < image_height:
            result_image.paste(image, (x, y))

    # 保存结果图片
    result_image.save(output_path)
