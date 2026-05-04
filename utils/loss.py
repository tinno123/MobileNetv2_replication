import torch
from torch import nn
import torch.nn.functional as F
from utils.labels import calculate_iou, build_labels
from utils.train_tools import GetGridCenter


# class YoloLoss(nn.Module):
#     def __init__(self, num_classes=80):
#         super(YoloLoss, self).__init__()
#         self.num_classes = num_classes
#         self.criterion = nn.BCEWithLogitsLoss()
#
#
#     def forward(self, pred, targets):
#         """
#
#         :param pred:  [N,num_classes]
#         :param targets: [N]
#         :return:
#         """
#         label = torch.zeros_like(pred)
#         label[torch.arange(pred.shape[0]).to(pred.device) , targets.int()] = 1
#         loss = F.cross_entropy(pred, label)
#
#         return loss
# class YoloLoss(nn.Module):
#     def __init__(self, num_classes=80):
#         super(YoloLoss, self).__init__()
#         self.num_classes = num_classes
#     def forward(self, pred, targets):
#         loss = F.cross_entropy(pred, targets.long())
#         return loss

class YoloLoss(nn.Module):
    def __init__(self, num_classes=80):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, pred, targets):
        loss = F.cross_entropy(pred, targets.long())

        # 计算正确率
        _, predicted = pred.max(1)  # 获取预测的类别
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total

        return loss,accuracy

