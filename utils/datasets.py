

import os.path

import cv2
import torch,torch.utils.data
from pathlib import Path
import random
import numpy as np
from PIL import Image

from utils.augmentation import image_augmentation


def rand( a=0, b=1):
    return np.random.rand() * (b - a) + a

# def get_random_data(image,box, input_shape, jitter=.1, hue=.1, sat=0.3, val=0.35, random=True):
#
#     # ------------------------------#
#     #   读取图像并转换成RGB图像
#     # ------------------------------#
#     image = image
#
#     # ------------------------------#
#     #   获得图像的高宽与目标高宽
#     # ------------------------------#
#     iw, ih = image.size
#     h, w = input_shape
#     # ------------------------------#
#     #   获得预测框
#     # ------------------------------#
#     box = box
#     if not random:
#         scale = min(w / iw, h / ih)
#         nw = int(iw * scale)
#         nh = int(ih * scale)
#         dx = (w - nw) // 2
#         dy = (h - nh) // 2
#
#         # ---------------------------------#
#         #   将图像多余的部分加上灰条
#         # ---------------------------------#
#         image = image.resize((nw, nh), Image.BICUBIC)
#         new_image = Image.new('RGB', (w, h), (128, 128, 128))
#         new_image.paste(image, (dx, dy))
#         image_data = np.array(new_image, np.float32)
#
#         # ---------------------------------#
#         #   对真实框进行调整
#         # ---------------------------------#
#         if len(box) > 0:
#             np.random.shuffle(box)
#             box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
#             box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
#             box[:, 0:2][box[:, 0:2] < 0] = 0
#             box[:, 2][box[:, 2] > w] = w
#             box[:, 3][box[:, 3] > h] = h
#             box_w = box[:, 2] - box[:, 0]
#             box_h = box[:, 3] - box[:, 1]
#             box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
#
#         return image_data, box
#
#     # ------------------------------------------#
#     #   对图像进行缩放并且进行长和宽的扭曲
#     # ------------------------------------------#
#     new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
#     scale = rand(0.95, 1.1)
#     if new_ar < 1:
#         nh = int(scale * h)
#         nw = int(nh * new_ar)
#     else:
#         nw = int(scale * w)
#         nh = int(nw / new_ar)
#     image = image.resize((nw, nh), Image.BICUBIC)
#
#     # ------------------------------------------#
#     #   将图像多余的部分加上灰条
#     # ------------------------------------------#
#     dx = int(rand(0, w - nw))
#     dy = int(rand(0, h - nh))
#     new_image = Image.new('RGB', (w, h), (128, 128, 128))
#     new_image.paste(image, (dx, dy))
#     image = new_image
#
#     # ------------------------------------------#
#     #   翻转图像
#     # ------------------------------------------#
#     flip = rand() < .5
#     # if flip:
#     #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
#
#     image_data = np.array(image, np.uint8)
#     # ---------------------------------#
#     #   对图像进行色域变换
#     #   计算色域变换的参数
#     # ---------------------------------#
#     r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
#     # ---------------------------------#
#     #   将图像转到HSV上
#     # ---------------------------------#
#     hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
#     dtype = image_data.dtype
#     # ---------------------------------#
#     #   应用变换
#     # ---------------------------------#
#     x = np.arange(0, 256, dtype=r.dtype)
#     lut_hue = ((x * r[0]) % 180).astype(dtype)
#     lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
#     lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
#
#     image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
#     image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
#
#     # ---------------------------------#
#     #   对真实框进行调整
#     # ---------------------------------#
#     if len(box) > 0:
#         np.random.shuffle(box)
#         box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
#         box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
#         if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
#         box[:, 0:2][box[:, 0:2] < 0] = 0
#         box[:, 2][box[:, 2] > w] = w
#         box[:, 3][box[:, 3] > h] = h
#         box_w = box[:, 2] - box[:, 0]
#         box_h = box[:, 3] - box[:, 1]
#         box = box[np.logical_and(box_w > 1, box_h > 1)]
#
#     return image_data, box


def get_random_data(image, box, input_shape, jitter=.15, hue=.25, sat=0.3,  val=0.7, random=True):
    # ------------------------------#
    #   读取图像并转换成RGB图像
    # ------------------------------#
    image = image

    # ------------------------------#
    #   获得图像的高宽与目标高宽
    # ------------------------------#
    iw, ih = image.size
    h, w = input_shape
    # ------------------------------#
    #   获得预测框
    # ------------------------------#
    box = box
    if not random:
        # 直接resize到目标尺寸，不保持宽高比
        image = image.resize((w, h), Image.BICUBIC)
        image_data = np.array(image, np.float32)

        # ---------------------------------#
        #   对真实框进行调整（直接缩放）
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * w / iw
            box[:, [1, 3]] = box[:, [1, 3]] * h / ih
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        return image_data, box

    # ------------------------------------------#
    #   对图像进行缩放并且进行长和宽的扭曲
    # ------------------------------------------#
    new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.7, 1.9)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)

    # 直接resize到计算出的尺寸
    image = image.resize((nw, nh), Image.BICUBIC)

    # ------------------------------------------#
    #   如果需要，可以随机裁剪或直接resize到目标尺寸
    #   这里改为直接resize到目标尺寸
    # ------------------------------------------#
    # 直接resize到目标尺寸，不进行裁剪和填充
    image = image.resize((w, h), Image.BICUBIC)
    image_data = np.array(image, np.uint8)

    # ------------------------------------------#
    #   翻转图像
    # ------------------------------------------#
    flip = rand() < .5
    # if flip:
    #     image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # ---------------------------------#
    #   对图像进行色域变换
    #   计算色域变换的参数
    # ---------------------------------#
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    # ---------------------------------#
    #   将图像转到HSV上
    # ---------------------------------#
    hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype = image_data.dtype
    # ---------------------------------#
    #   应用变换
    # ---------------------------------#
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

    # ---------------------------------#
    #   对真实框进行调整（直接缩放）
    # ---------------------------------#
    if len(box) > 0:
        np.random.shuffle(box)
        # 直接缩放，不加偏移量
        box[:, [0, 2]] = box[:, [0, 2]] * w / iw
        box[:, [1, 3]] = box[:, [1, 3]] * h / ih
        if flip:
            box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]

    return image_data, box


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir : str, label_dir : str ,images_dir: str,anchors,image_size,mode = 'train'):

        #----------------------#
        # 标签路径
        #----------------------#
        self.labelpath =  Path(root_dir) /  Path( label_dir)

        # ----------------------#
        # 图像路径
        # ----------------------#
        self.imagespath = Path(root_dir) /  Path(images_dir)

        # ----------------------#
        # 模型输入尺寸
        # ----------------------#
        self.image_size= image_size

        # ----------------------#
        # 标签构建所需的Anchors
        # ----------------------#
        self.anchors = anchors

        # ----------------------#
        # 标签构建模式
        # ----------------------#
        self.mode = mode

        # ----------------------#
        # 加载标签
        # ----------------------#
        self.samples = []
        self.load_samples()

        # ----------------------#
        # 图像增强
        # ----------------------#
        self.augmentation = image_augmentation

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]

        # -------------------------#
        # 获取当前batch所需的图像和标签
        # -------------------------#
        image = Image.open(sample['image_path']).convert('RGB')
        class_id = self.analysis_label(sample['label_path'])

        # -------------------------#
        # 获取当前batch图像尺寸
        # -------------------------#
        w, h = image.size

        # -------------------------#
        # 进行图像增强
        # -------------------------#
        bboxes = []
        if self.mode == 'train':
            #image, bboxes = image_augmentation(image, bboxes, self.image_size)
            w, h = image.size
            input_shape = self.image_size
            image, bboxes = get_random_data(image, bboxes, input_shape, random=True)
        else:
            input_shape = self.image_size
            image, bboxes = get_random_data(image, bboxes, input_shape, random=False)

        #

        # -------------------------#
        # ToTensor
        # -------------------------#
        image_tensor = self.image2tensor(image)
        classes_tensor = self.bboxes2tensor(class_id)
        self._save_vis(image_tensor, sample['image_path'])





        return image_tensor ,classes_tensor

    def image2tensor(self, image, normalize='imagenet'):
        """
        Args:
            image: 输入图像
            normalize:
                - False/None: 不标准化
                - 'imagenet': 使用ImageNet参数
                - 'custom': 需要同时传入mean,std
                - 'zero_one': 归一化到[-1,1]
        """
        # 转换到0-1范围
        if isinstance(image, Image.Image):
            image = np.array(image)

        tensor = torch.from_numpy(image.astype(np.float32)) / 255.0
        tensor = tensor.permute(2, 0, 1).contiguous()

        # 标准化
        if normalize == 'imagenet':
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std

        elif normalize == 'zero_one':
            # 归一化到[-1, 1]
            tensor = tensor * 2 - 1

        elif isinstance(normalize, dict):
            mean = torch.tensor(normalize['mean']).view(3, 1, 1)
            std = torch.tensor(normalize['std']).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor

    def bboxes2tensor(self, bboxes):
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32) if bboxes.shape[0] > 0 else torch.zeros((0, 1))
        return bboxes_tensor



    def load_samples(self):
        all_files = os.listdir(self.labelpath)
        for file in all_files:
            labelpath = Path(self.labelpath / Path(file))

            if os.path.isfile(labelpath) and file.endswith('.txt'):
                imagepath = Path( self.imagespath / Path(file.split(".")[0] + ".jpg"))
                if os.path.isfile(imagepath):
                    self.samples.append({
                            'image_path': imagepath,
                            'label_path': labelpath
                    })
                else:
                    print(f"Warning: Image file {imagepath} not found, skipping...")

    def analysis_label(self, label_path):
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                bboxes.append(int(parts[0]))

        return  np.array(bboxes)



    # def _save_vis(self, img_tensor, src_path, tgt_size=224, save_dir='runs/vis'):
    #     """
    #     把 tensor 画框后存成 jpg，方便肉眼检查坐标是否正确。
    #     输入：
    #         img_tensor : C×H×W  0-1
    #         bboxes     : [[cls,cx,cy,w,h], ...]  归一化
    #         src_path   : 原图路径（仅用于命名）
    #         tgt_size   : 画布尺寸
    #     """
    #     import os, cv2
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     # tensor -> numpy RGB
    #     img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #
    #     save_name = os.path.basename(src_path).replace('.jpg', '_vis.jpg')
    #     from pathlib import Path
    #     save_path = Path(save_dir) / save_name
    #     cv2.imwrite(str(save_path), img)
    def _save_vis(self, img_tensor, src_path, tgt_size=224, save_dir='runs/vis'):
        """
        把 tensor 画框后存成 jpg，方便肉眼检查坐标是否正确。
        输入：
            img_tensor : C×H×W  ImageNet归一化后的tensor
            src_path   : 原图路径（仅用于命名）
            tgt_size   : 画布尺寸
        """
        import os, cv2
        os.makedirs(save_dir, exist_ok=True)

        # 反归一化（ImageNet参数）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # 反归一化：x = (x * std) + mean
        img_denorm = img_tensor * std + mean
        # 限制范围在[0, 1]
        img_denorm = torch.clamp(img_denorm, 0, 1)

        # tensor -> numpy RGB
        img = (img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        save_name = os.path.basename(src_path).replace('.jpg', '_vis.jpg')
        from pathlib import Path
        save_path = Path(save_dir) / save_name
        cv2.imwrite(str(save_path), img)