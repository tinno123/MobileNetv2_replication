import math
import random

import cv2
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import collections
import torch
from utils.datasets import Dataset


class CosineDecayLR(object):
    def __init__(self, max_lr, min_lr, total_epochs,warmup_epochs=0.01):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = int(warmup_epochs * total_epochs)
        self.remain_epoch = self.total_epochs - self.warmup_epochs

    def warmup(self,epoch):
        if type(self.max_lr) == list:
            lr = []
            for i in self.max_lr:
                lr.append(i* epoch / self.warmup_epochs)
            return lr
        else:
            return self.max_lr * epoch / self.warmup_epochs

    def cosine_decay(self,epoch):
        current_epoch = epoch - self.warmup_epochs
        theta = (1 - current_epoch / self.remain_epoch) * math.pi

        if type(self.max_lr) == list:
            lr = []
            for maxlr , minlr in zip(self.max_lr,self.min_lr):
                lr.append(maxlr - (maxlr - minlr) * (1 + math.cos(theta)) / 2)

        else:
            lr = self.max_lr - (self.max_lr - self.min_lr) * (1 + math.cos(theta)) / 2
        return lr

    def update_lr(self,optimizer,lr):
        if type(lr) == list and type(optimizer) == list:
            for o,lr in zip(optimizer,lr):
                for param_group in o.param_groups:
                    param_group['lr'] = lr
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def __call__(self, epoch,optimizer):

        epoch = min(epoch + 1, self.total_epochs)

        #----------------------------#
        # 热身阶段,LR线性上升
        # ---------------------------#
        if epoch <= self.warmup_epochs:
            lr = self.warmup(epoch)
        else:
        # ----------------------------#
        # 余弦退火阶段，从热身之后开始
        # ----------------------------#
            lr = self.cosine_decay(epoch)

        # ----------------------------#
        # 更新优化器的学习率
        # ----------------------------#
        if optimizer is not None:
            self.update_lr(optimizer,lr)

        return lr

def GetOptimizer(model,optimizer_type,lr,momentum,weight_decay):
    pg_normal_weight, pg_bias, pg_bn_weight = [], [], []
    for k , v  in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg_bias.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg_bn_weight.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg_normal_weight.append(v.weight)
    optimizer = {
        'adam':
            optim.Adam([
                {'params': pg_normal_weight, 'weight_decay': weight_decay},
                {'params': pg_bias},
                {'params': pg_bn_weight}
            ], lr, betas=(momentum, 0.999)),
        'sgd':
            optim.SGD([
                {'params': pg_normal_weight, 'weight_decay': weight_decay},
                {'params': pg_bias},
                {'params': pg_bn_weight}
            ], lr, momentum=momentum, nesterov=True)
    }[optimizer_type]
    return optimizer


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def GetLoader(
        root_dir,
        train_annotation_path,
        train_image_path,
        val_annotation_path,
        val_image_path,
        batch_size,
        num_workers,
        input_shape

):
    anchors = torch.tensor([
        [116, 90], [156, 198], [373, 326],
        [30, 61], [62, 45], [59, 119],
        [10, 13], [16, 30], [33, 23]
    ])

    train_dataset = Dataset(root_dir,
                            train_annotation_path,
                            train_image_path,
                            anchors,
                            input_shape,
                            mode="train"
                            )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, persistent_workers=True,
                         drop_last=True)
    val_dataset = Dataset(root_dir,
                          val_annotation_path,
                          val_image_path,
                          anchors,
                          input_shape,
                          mode="val"
                          )
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True, persistent_workers=True,
                            drop_last=False)
    return train_loader, val_loader


def GetGridCenter(feature_map_size,num_anchors_perscale,stride,mode ="train"):
    w, h = feature_map_size
    grid_x = torch.arange(w, dtype=torch.float32)
    grid_y = torch.arange(h, dtype=torch.float32)
    grid_x = grid_x.repeat(h)
    grid_y = grid_y.repeat_interleave(w)
    if mode == "train":
        origin_center = torch.stack([(grid_x + 0.5) * stride, (grid_y + 0.5) * stride], dim=-1)
    else :
        origin_center = torch.stack([grid_x, grid_y], dim=-1)
    return origin_center.repeat_interleave(num_anchors_perscale, dim=0)

def image2tensor(image: np.ndarray,device,image_size = [416,416]):
    image = cv2.resize(image, (image_size[0], image_size[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)
    image = image / 255.0
    return image.unsqueeze(0).to(device)


def load_weights_by_shape(model, weight_file):
    """
    此方法针引用模型权重键名不一致且确定结构一致
    :param model:
    :param weight_file:
    :return:
    """
    # -----------------------
    # 加载权重
    # -----------------------
    weights_values = []

    if isinstance(weight_file, str):
        try:
            weights = torch.load(weight_file)
            for k, v in weights.items():
                if "num_batches_tracked" in k:
                    continue
                weights_values.append(v)
        except Exception as e:
            raise ValueError(f"加载模型文件失败：{e}")
    elif isinstance(weight_file, (dict, collections.OrderedDict)):
        weights = weight_file
        for k, v in weights.items():
            if "num_batches_tracked" in k:
                continue
            weights_values.append(v)
    else:
        raise ValueError(f"不支持的权重类型：{type(weight_file)}")

    if len(weights_values) == 0:
        raise ValueError("权重文件为空或格式不正确")
    # -----------------------
    # 模型权重赋值
    # -----------------------
    index = 0
    success_num = 0
    fail = []
    model_state_dict = model.state_dict()
    for k, v in model_state_dict.items():
        if index >= len(weights_values):
            break
        # -----------------------
        # 去除批次数量追踪层
        # -----------------------
        if "num_batches_tracked" in k:
            continue
        if weights_values[index].shape == v.shape:
            model_state_dict[k] = weights_values[index]
            success_num += 1
        else:
            print(f"权重形状不一致: {k}")
            fail.append((k, v.shape, weights_values[index].shape))
        index += 1
    print(f"加载数量 / 权重总量 : {success_num} / {len(weights_values)}" )
    print(f"加载失败:{fail}")

    model.load_state_dict(model_state_dict)






def load_by_shape_precise(source_path: str, target_model: torch.nn.Module,
                          verbose: bool = True):
    """
    按形状精确加载预训练权重

    Args:
        source_path: 源模型文件路径，可以是包含state_dict的字典或直接是state_dict
        target_model: 目标模型
        verbose: 是否显示详细信息
    """
    # 加载源模型文件
    try:
        source_data = torch.load(source_path, weights_only=True)
    except Exception as e:
        raise ValueError(f"加载模型文件失败: {e}")

    # 检查文件格式并提取state_dict
    if isinstance(source_data, dict):
        # 如果是字典格式，检查是否包含常见的键
        if 'state_dict' in source_data:
            # 格式: {'state_dict': ..., 'epoch': ..., 'optimizer': ...}
            source_state = source_data['state_dict']
            if verbose:
                print("📁 检测到完整模型字典格式，提取state_dict")
                if 'epoch' in source_data:
                    print(f"📅 模型训练轮次: {source_data.get('epoch', 'N/A')}")
        elif 'model' in source_data:
            # 格式: {'model': ..., 'other_info': ...}
            source_state = source_data['model']
            if verbose:
                print("📁 检测到模型字典格式，提取model")
        else:
            # 如果字典中不包含常见键，假设整个字典就是state_dict
            source_state = source_data
            if verbose:
                print("📁 检测到state_dict字典格式")
    elif isinstance(source_data, collections.OrderedDict):
        # 如果是OrderedDict，直接作为state_dict
        source_state = source_data
        if verbose:
            print("📁 检测到OrderedDict格式(state_dict)")
    else:
        raise ValueError(f"不支持的模型文件格式: {type(source_data)}")

    if verbose:
        print(f"📊 源state_dict键数量: {len(source_state)}")

    # 过滤掉 num_batches_tracked，不加载这些参数
    source_tensors = {k: v for k, v in source_state.items()
                      if isinstance(v, torch.Tensor) and 'num_batches_tracked' not in k}

    # 获取目标模型状态
    target_state = target_model.state_dict()

    # 按形状分组源张量
    shape_to_sources = collections.defaultdict(list)
    for key, tensor in source_tensors.items():
        shape_to_sources[tensor.shape].append((key, tensor))

    if verbose:
        print("\n=== 源模型形状统计 ===")
        shape_stats = {shape: len(tensors) for shape, tensors in shape_to_sources.items()}
        for shape, count in shape_stats.items():
            print(f"形状 {shape}: {count}个张量")

    # 创建新状态字典
    new_state_dict = collections.OrderedDict()
    matched_count = 0
    unused_sources = list(source_tensors.keys())

    # 逐层匹配目标模型的每个参数
    for target_key, target_param in target_state.items():
        # 如果是 num_batches_tracked，直接使用目标模型的初始值
        if 'num_batches_tracked' in target_key:
            new_state_dict[target_key] = target_param
            if verbose:
                print(f"🔷 跳过: {target_key} (使用初始值)")
            continue

        target_shape = target_param.shape

        if target_shape in shape_to_sources and shape_to_sources[target_shape]:
            # 找到形状匹配的源张量
            source_key, source_tensor = shape_to_sources[target_shape].pop(0)
            new_state_dict[target_key] = source_tensor
            matched_count += 1
            if source_key in unused_sources:
                unused_sources.remove(source_key)

            if verbose:
                print(f"✅ 形状匹配: {source_key} -> {target_key}")
        else:
            # 保持目标模型的初始权重
            new_state_dict[target_key] = target_param
            if verbose:
                print(f"⚠️  形状不匹配: {target_key} (使用初始权重)")

    # 统计信息
    if verbose:
        print(f"\n=== 加载统计 ===")
        print(f"目标模型总层数: {len(target_state)}")
        print(f"成功匹配层数: {matched_count}")
        print(f"匹配率: {matched_count / len(target_state) * 100:.1f}%")
        print(f"未使用的源张量: {len(unused_sources)}")
        if unused_sources:
            print("未使用的源张量:")
            for key in unused_sources[:10]:  # 只显示前10个
                print(f"  - {key}")
            if len(unused_sources) > 10:
                print(f"  ... 还有{len(unused_sources) - 10}个未显示")

    # 确保加载状态字典并返回模型
    target_model.load_state_dict(new_state_dict)
    return target_model








if __name__ == '__main__':
    lr_scheduler = CosineDecayLR(max_lr=0.1, min_lr=0.001, total_epochs=100)
    for epoch in range(100):
        lr = lr_scheduler(epoch,None)
        print(lr)
    GetGridCenter([13,13],3,32,"detect")





