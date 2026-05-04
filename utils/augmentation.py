import random
import numpy as np
from PIL import Image
import cv2

def image_augmentation(image, bboxes,target_shape,scale_range=(0.3,1.0),hue=0.05,sat=0.3,val=0.2,p = True):
    """
    :param image: PIL图像
    :param bboxes: np.array 的标签，shape = (N,5) [class_id, xc,yc,w,h](归一化)
    :param target_shape: 训练要求的图像大小
    :param hue:   色相
    :param sat:   饱和度
    :param val:   亮度
    :param p:     是否开启几何+颜色变换
    :return:      变换后的图像和标签  image(np)、bboxes(np)(归一化)
    """
    # ------------------------------#
    #   获得图像的高宽与目标高宽
    # ------------------------------#
    origen_w, origen_h = image.size
    target_w, target_h = target_shape

    # ------------------------------#
    #   反归一化,方便计算
    # ------------------------------#
    bboxes = bboxes.copy()
    bboxes[:, [1,3]]  = bboxes[:, [1,3]] * origen_w
    bboxes[:, [2,4]]  = bboxes[:, [2,4]] * origen_h

    if  p:
        # ------------------------------------------------------------#
        #   调整输入图片大小，适配输出的图片，这一步这是为了保证图片一定在训练范围内
        #   1.resize:直接拉伸到目标尺寸,强行改变尺寸
        #   2.letterbox:保持宽高比不变，为了保证图片不变形并且在目标尺寸内
        # -------------------------------------------------------------#
        if random.random() < 0.8:
            # 直接拉伸
            image = image.resize((target_w, target_h), Image.BICUBIC)
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * target_w / origen_w
            bboxes[:, [2, 4]] = bboxes[:, [2, 4]] * target_h / origen_h
        else:
            # letterbox
            letterbox_scale = min(target_w / origen_w, target_h / origen_h)
            new_w, new_h = int(origen_w * letterbox_scale), int(origen_h * letterbox_scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            # 这里要进行一个缩放
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * letterbox_scale
            bboxes[:, [2, 4]] = bboxes[:, [2, 4]] * letterbox_scale


        new_image = Image.new('RGB', (target_w, target_h), (128, 128, 128))


        # ------------------------------#
        #  图像缩放
        # ------------------------------#
        random_scale = random.uniform(*scale_range) # 随机缩放
        new_w,new_h = int(image.size[0]*random_scale), int(image.size[1]*random_scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)

        # ------------------------------#
        #  图像随机放置
        # ------------------------------#
        random_left = random.randint(0, target_w - new_w)
        random_top = random.randint(0, target_h - new_h)
        new_image.paste(image, (random_left, random_top))

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        bboxes[:,[1,3]] = bboxes[:,[1,3]] * random_scale #调整宽度和中心点x
        bboxes[:,[2,4]] = bboxes[:,[2,4]] * random_scale #调整高度和中心点y
        bboxes[:, 1] = bboxes[:, 1] + random_left        #调整中心点x偏移
        bboxes[:, 2] = bboxes[:, 2] + random_top         #调整中心点y偏移


        # ------------------------------#
        #  图像翻转
        # ------------------------------#
        if random.random() < 0.5:
            new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
            bboxes[:,1] = target_w/2 + (target_w/2-bboxes[:,1]) #使用对称公式进行翻转

        # ---------------------------------#
        #   颜色变换
        # ---------------------------------#
        image = np.array(new_image, np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h,s,v = cv2.split(hsv)
        h_scale = random.uniform(-hue, hue)
        s_scale = random.uniform(-sat, sat)
        v_scale = random.uniform(-val, val)
        h = np.clip((h*(1+h_scale)), 0, 180)
        s = np.clip((s*(1+s_scale)), 0, 255)
        v = np.clip((v*(1+v_scale)), 0, 255)
        hsv = cv2.merge([h,s,v]).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # 归一化
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / target_w
        bboxes[:, [2, 4]] = bboxes[:, [2, 4]] / target_h
        bboxes[:,1:5] = np.clip(bboxes[:,1:5], 0, 1)

        return image, bboxes
    else:
        letterbox_scale = min(target_w / origen_w, target_h / origen_h)
        new_w, new_h = int(origen_w * letterbox_scale), int(origen_h * letterbox_scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)

        new_image = Image.new('RGB', (target_w, target_h), (128, 128, 128))
        new_image.paste(image, (int((target_w - new_w) / 2), int((target_h - new_h) / 2)))

        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * letterbox_scale  # 调整宽度和中心点x
        bboxes[:, [2, 4]] = bboxes[:, [2, 4]] * letterbox_scale  # 调整高度和中心点y
        bboxes[:, 1] = bboxes[:, 1] + int((target_w - new_w) / 2)
        bboxes[:, 2] = bboxes[:, 2] + int((target_h - new_h) / 2)

        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / target_w
        bboxes[:, [2, 4]] = bboxes[:, [2, 4]] / target_h

        return np.array(new_image), bboxes


# def image_augmentation(image, bboxes, target_shape, scale_range=(0.3, 1.0), hue=0.05, sat=0.3, val=0.2, p=True):
#     """
#     :param image: PIL图像（必传，参数不变）
#     :param bboxes: np.array 的标签，shape = (N,5) [class_id, xc,yc,w,h](归一化)（必传，参数不变）
#     :param target_shape: 训练要求的图像大小 (w, h)（必传，参数不变）
#     :param scale_range: 兼容原有参数（内部已替换为更优的缩放逻辑，此参数仅占位）
#     :param hue: 兼容原有参数（内部放大为工业界标准幅度）
#     :param val: 兼容原有参数（内部放大为工业界标准幅度）
#     :param sat: 兼容原有参数（内部放大为工业界标准幅度）
#     :param p: 是否开启增强（参数保留，逻辑优化）
#     :return: 变换后的图像(np)、bboxes(np)(归一化)（输出格式不变）
#     """
#     # ------------------------------#
#     #   基础参数获取
#     # ------------------------------#
#     origen_w, origen_h = image.size
#     target_w, target_h = target_shape
#     # 深拷贝避免修改原数据
#     bboxes = bboxes.copy()
#     # 反归一化：将归一化的框转回像素坐标（方便几何变换计算）
#     if len(bboxes) > 0:
#         bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * origen_w  # xc, w → 像素
#         bboxes[:, [2, 4]] = bboxes[:, [2, 4]] * origen_h  # yc, h → 像素
#
#     # ------------------------------#
#     #   不增强时：仅做letterbox填充（保持原逻辑兼容性）
#     # ------------------------------#
#     if not p:
#         letterbox_scale = min(target_w / origen_w, target_h / origen_h)
#         new_w, new_h = int(origen_w * letterbox_scale), int(origen_h * letterbox_scale)
#         image = image.resize((new_w, new_h), Image.BICUBIC)
#
#         new_image = Image.new('RGB', (target_w, target_h), (128, 128, 128))
#         dx = (target_w - new_w) // 2
#         dy = (target_h - new_h) // 2
#         new_image.paste(image, (dx, dy))
#
#         # 调整框坐标
#         if len(bboxes) > 0:
#             bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * letterbox_scale + dx
#             bboxes[:, [2, 4]] = bboxes[:, [2, 4]] * letterbox_scale + dy
#
#         # 归一化回[0,1]范围
#         bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / target_w
#         bboxes[:, [2, 4]] = bboxes[:, [2, 4]] / target_h
#         return np.array(new_image, np.uint8), bboxes
#
#     # ------------------------------#
#     #   增强逻辑（核心替换为get_random_data的优质策略）
#     # ------------------------------#
#     # 1. 随机长宽比扭曲（模拟不同拍摄角度，避免目标变形）
#     jitter = 0.3  # 工业界YOLO标准值
#     new_ar = (origen_w / origen_h) * random.uniform(1 - jitter, 1 + jitter) / random.uniform(1 - jitter, 1 + jitter)
#
#     # 2. 宽范围缩放（0.25~2.0倍，覆盖“远小近大”真实场景，弃用原scale_range）
#     scale = random.uniform(0.25, 2.0)
#     if new_ar < 1:
#         nh = int(scale * target_h)
#         nw = int(nh * new_ar)
#     else:
#         nw = int(scale * target_w)
#         nh = int(nw / new_ar)
#     image = image.resize((nw, nh), Image.BICUBIC)
#
#     # 3. 随机平移放置（模拟目标在画面中不同位置）
#     dx = random.randint(0, max(0, target_w - nw))
#     dy = random.randint(0, max(0, target_h - nh))
#     new_image = Image.new('RGB', (target_w, target_h), (128, 128, 128))
#     new_image.paste(image, (dx, dy))
#
#     # 4. 调整框坐标（缩放+平移）
#     if len(bboxes) > 0:
#         # 缩放框
#         bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * (nw / origen_w)
#         bboxes[:, [2, 4]] = bboxes[:, [2, 4]] * (nh / origen_h)
#         # 平移框（中心点）
#         bboxes[:, 1] += dx
#         bboxes[:, 2] += dy
#
#     # 5. 随机水平翻转（修正框坐标，适配中心点格式）
#     flip = random.random() < 0.5
#     if flip and len(bboxes) > 0:
#         new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
#         # 翻转后中心点x坐标修正：target_w - 原x坐标
#         bboxes[:, 1] = target_w - bboxes[:, 1]
#
#     # 6. 颜色变换（改用LUT查找表，更自然，放大变换幅度到工业界标准）
#     image_data = np.array(new_image, np.uint8)
#     # 放大原参数幅度（适配真实场景的光照/色彩变化）
#     hue = min(0.1, hue * 2)  # 原hue=0.05 → 实际用0.1
#     sat = min(0.7, sat * 2.3)  # 原sat=0.3 → 实际用0.7
#     val = min(0.4, val * 2)  # 原val=0.2 → 实际用0.4
#
#     # HSV色域变换（LUT方式更稳定）
#     hsv = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
#     h, s, v = cv2.split(hsv)
#     dtype = image_data.dtype
#     # 生成变换参数
#     r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
#     # 应用LUT变换（避免直接乘系数导致的色彩失真）
#     x = np.arange(0, 256, dtype=r.dtype)
#     lut_hue = ((x * r[0]) % 180).astype(dtype)
#     lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
#     lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
#     # 合并通道并转回RGB
#     hsv = cv2.merge((cv2.LUT(h, lut_hue), cv2.LUT(s, lut_sat), cv2.LUT(v, lut_val)))
#     image_data = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#
#     # 7. 过滤无效框+坐标裁剪（避免训练噪声）
#     if len(bboxes) > 0:
#         # 裁剪坐标到图像范围内
#         bboxes[:, 1] = np.clip(bboxes[:, 1], 0, target_w)
#         bboxes[:, 2] = np.clip(bboxes[:, 2], 0, target_h)
#         bboxes[:, 3] = np.clip(bboxes[:, 3], 1, target_w)  # 宽度至少1像素
#         bboxes[:, 4] = np.clip(bboxes[:, 4], 1, target_h)  # 高度至少1像素
#         # 过滤宽/高<1的无效框
#         box_w = bboxes[:, 3]
#         box_h = bboxes[:, 4]
#         bboxes = bboxes[np.logical_and(box_w > 1, box_h > 1)]
#
#     # 8. 归一化回[0,1]范围（保持输出格式不变）
#     bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / target_w
#     bboxes[:, [2, 4]] = bboxes[:, [2, 4]] / target_h
#     bboxes[:, 1:5] = np.clip(bboxes[:, 1:5], 0, 0.99)
#
#     return image_data, bboxes

